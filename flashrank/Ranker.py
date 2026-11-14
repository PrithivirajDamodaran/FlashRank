import json
from pathlib import Path
from tokenizers import AddedToken, Tokenizer
import onnxruntime as ort
import numpy as np
import os
import zipfile
import requests
from tqdm import tqdm
from flashrank.Config import (
    default_model,
    default_cache_dir,
    model_url,
    model_file_map,
    listwise_rankers,
    hf_model_url,
    required_files,
    optional_files,
    onnx_file_candidates,
)
import collections
from typing import Optional, List, Dict, Any, Set
import logging
from functools import cached_property

logger = logging.getLogger(__name__)

def _format_size(num_bytes: int) -> str:
    """ Formats a byte count for log output. """
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"

def download_file(local_file: Path, download_url: str) -> None:
    """ Downloads a file to a local path, writing it atomically.

    The payload is streamed to a sibling ``.part`` file and only moved into
    place once the transfer completes. An interrupted download therefore
    leaves no truncated file that would look complete on the next run.

    Args:
        local_file (Path): Destination path for the downloaded file.
        download_url (str): URL to download from.
    """
    partial_file = local_file.with_name(f"{local_file.name}.{os.getpid()}.part")
    try:
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            if total_size:
                logger.info(f"Downloading {download_url} ({_format_size(total_size)})")
            with open(partial_file, 'wb') as f, tqdm(desc=local_file.name, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
        os.replace(partial_file, local_file)
    except BaseException:
        if partial_file.exists():
            partial_file.unlink()
        raise

class RerankRequest:
    """ Represents a reranking request with a query and a list of passages. 
    
    Attributes:
        query (Optional[str]): The query for which the passages need to be reranked.
        passages (List[Dict[str, Any]]): The list of passages to be reranked.
    """

    def __init__(self, query: Optional[str] = None, passages: Optional[List[Dict[str, Any]]] = None):
        self.query: Optional[str] = query
        self.passages: List[Dict[str, Any]] = passages if passages is not None else []

class Ranker:
    """ A ranker class for reranking passages based on a provided query using a pre-trained model.

    Attributes:
        cache_dir (Path): Path to the cache directory where models are stored.
        model_dir (Path): Path to the directory of the specific model being used.
        session (ort.InferenceSession): The ONNX runtime session for making inferences.
        tokenizer (Tokenizer): The tokenizer for text processing.
    """

    def __init__(self, model_name: str = default_model, cache_dir: str = default_cache_dir, max_length: int = 512, log_level: str = "INFO"):
        """ Initializes the Ranker class with specified model and cache settings.

        Args:
            model_name (str): The name of the model to be used.
            cache_dir (str): The directory where models are cached.
            max_length (int): The maximum length of the tokens.
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        
        # Setting up logging
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
        self.logger = logging.getLogger(__name__)

        self.cache_dir: Path = Path(cache_dir)
        self.model_dir: Path = self.cache_dir / model_name.replace("/", "-")
        if model_name in model_file_map:
            model_file = model_file_map[model_name]
        else:
            model_file = f"{model_name.split('/')[-1]}.onnx"
        self.model_path = self.model_dir / model_file
        self._prepare_model_dir(model_name)

        self.llm_model = None
        if model_name in listwise_rankers:
            try:
                from llama_cpp import Llama
                self.llm_model = Llama(
                model_path=str(self.model_path),
                n_ctx=max_length,  
                n_threads=8,          
                ) 
            except ImportError:
                raise ImportError("Please install it using 'pip install flashrank[listwise]' to run LLM based listwise rerankers.")    
        else:
            self.session = ort.InferenceSession(str(self.model_path))
            self.tokenizer: Tokenizer = self._get_tokenizer(max_length)

    @cached_property
    def input_names(self) -> Set[str]:
        """ Names of the inputs the loaded ONNX graph declares.

        Computed once, on first use, so it is not rebuilt on every rerank call.
        """
        return {inp.name for inp in self.session.get_inputs()}

    def _is_model_ready(self, model_name: str) -> bool:
        """ Returns True when every file the ranker needs is already cached.

        Curated models are complete once their model file is present. Custom Hub
        models also need their tokenizer files: the model file is downloaded
        first, so a later failure would otherwise leave a cache directory that
        looks ready but cannot load, and it would never be retried.
        """
        if not self.model_path.exists():
            return False
        if model_name in model_file_map:
            return True
        return all((self.model_dir / name).exists() for name in required_files)

    def _prepare_model_dir(self, model_name: str):
        """ Ensures the model directory is prepared by downloading and extracting the model if not present.

        Args:
            model_name (str): The name of the model to be prepared.
        """
        if not self.cache_dir.exists():
            self.logger.debug(f"Cache directory {self.cache_dir} not found. Creating it..")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not self._is_model_ready(model_name):
            self.logger.info(f"Downloading {model_name}...")
            if model_name in model_file_map:
                self._download_model_files(model_name)
            else:
                self.logger.info(f"Model {model_name} not found in model map, downloading from custom URL...")
                self._download_hf_model_files(model_name)

    def _download_model_files(self, model_name: str):
        """ Downloads and extracts the model files from a specified URL.

        Args:
            model_name (str): The name of the model to download.
        """
        local_zip_file = self.cache_dir / f"{model_name}.zip"
        formatted_model_url = model_url.format(model_name)
        
        download_file(local_zip_file, formatted_model_url)

        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.cache_dir)
        os.remove(local_zip_file)

    def _download_onnx_variant(self, model_name: str, local_model_path: Path):
        """ Downloads the first ONNX variant of a Hub model that exists.

        Args:
            model_name (str): The name of the model on the Hub.
            local_model_path (Path): Where to store the downloaded ONNX file.
        """
        for onnx in onnx_file_candidates:
            formatted_model_url = hf_model_url.format(model_name, onnx)
            try:
                download_file(local_model_path, formatted_model_url)
                return
            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status in (401, 403):
                    raise PermissionError(
                        f"Model {model_name} is gated or private on the Hugging Face Hub "
                        f"(HTTP {status}). Accept its terms on the Hub before using it."
                    ) from e
                self.logger.debug(f"Could not download {onnx} for model {model_name}: {e}")
        raise FileNotFoundError(
            f"No ONNX model found for {model_name}. Looked for: {', '.join(onnx_file_candidates)}."
        )

    def _download_hf_model_files(self, model_name: str):
        """ Downloads model files from Hugging Face repository.

        Files already present are left alone, so a download that failed partway
        can be resumed without re-fetching the model.

        Args:
            model_name (str): The name of the model to download.
        """
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
        local_model_path = self.model_dir / f"{model_name.split('/')[-1]}.onnx"
        if not local_model_path.exists():
            self._download_onnx_variant(model_name, local_model_path)

        for req_file in required_files:
            local_file_path = self.model_dir / req_file
            if not local_file_path.exists():
                download_file(local_file_path, hf_model_url.format(model_name, req_file))

        for opt_file in optional_files:
            local_file_path = self.model_dir / opt_file
            if local_file_path.exists():
                continue
            try:
                download_file(local_file_path, hf_model_url.format(model_name, opt_file))
            except requests.exceptions.HTTPError as e:
                self.logger.debug(f"Optional file {opt_file} not available for {model_name}: {e}")

    def _get_tokenizer(self, max_length: int = 512) -> Tokenizer:
        """ Initializes and configures the tokenizer with padding and truncation.

        Args:
            max_length (int): The maximum token length for truncation.

        Returns:
            Tokenizer: Configured tokenizer for text processing.
        """
        with open(str(self.model_dir / "config.json")) as config_file:
            config = json.load(config_file)
        with open(str(self.model_dir / "tokenizer_config.json")) as tokenizer_config_file:
            tokenizer_config = json.load(tokenizer_config_file)
        tokens_map = {}
        tokens_map_path = self.model_dir / "special_tokens_map.json"
        if tokens_map_path.exists():
            with open(str(tokens_map_path)) as tokens_map_file:
                tokens_map = json.load(tokens_map_file)
        tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))

        # Not every Hub repo populates these, so fall back rather than raising.
        try:
            model_max_length = int(tokenizer_config.get("model_max_length") or max_length)
        except (TypeError, ValueError):
            model_max_length = max_length
        tokenizer.enable_truncation(max_length=min(model_max_length, max_length))
        pad_token = tokenizer_config.get("pad_token")
        pad_id = config.get("pad_token_id")
        if pad_token is not None:
            tokenizer.enable_padding(pad_id=pad_id or 0, pad_token=pad_token)
        else:
            tokenizer.enable_padding(pad_id=pad_id or 0)

        for token in tokens_map.values():
            if isinstance(token, str):
                tokenizer.add_special_tokens([token])
            elif isinstance(token, dict):
                tokenizer.add_special_tokens([AddedToken(**token)])

        vocab_file = self.model_dir / "vocab.txt"
        if vocab_file.exists():
            tokenizer.vocab = self._load_vocab(vocab_file)
            tokenizer.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in tokenizer.vocab.items()])
        return tokenizer

    def _load_vocab(self, vocab_file: Path) -> Dict[str, int]:
        """ Loads the vocabulary from a file and returns it as an ordered dictionary.

        Args:
            vocab_file (Path): The file path to the vocabulary.

        Returns:
            Dict[str, int]: An ordered dictionary mapping tokens to their respective indices.
        """
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab
    
    def _get_prefix_prompt(self, query, num):
        return [
            {
                "role": "system",
                "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.",
            },
            {
                "role": "user",
                "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
            },
            {"role": "assistant", "content": "Okay, please provide the passages."},
        ]

    def _get_postfix_prompt(self, query, num):
        example_ordering = "[2] > [1]"
        return {
            "role": "user",
            "content": f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}, Only respond with the ranking results, do not say any word or explain.",
        }



    def rerank(self, request: RerankRequest) -> List[Dict[str, Any]]:
        """ Reranks a list of passages based on a query using a pre-trained model.

        Args:
            request (RerankRequest): The request containing the query and passages to rerank.

        Returns:
            List[Dict[str, Any]]: The reranked list of passages with added scores.
        """
        query = request.query
        passages = request.passages

        # self.llm_model will be instantiated for GGUF based Listwise LLM models
        if self.llm_model is not None:
            self.logger.debug("Running listwise ranking..")
            num_of_passages = len(passages)
            messages = self._get_prefix_prompt(query, num_of_passages)

            result_map = {}
            for rank, passage in enumerate(passages):
                messages.append(
                    {
                        "role": "user",
                        "content": f"[{rank + 1}] {passage['text']}",
                    }
                )
                messages.append(
                        {
                            "role": "assistant", 
                            "content": f"Received passage [{rank + 1}]."
                        }
                )
                
                result_map[rank + 1] = passage

            messages.append(self._get_postfix_prompt(query, num_of_passages))
            raw_ranks = self.llm_model.create_chat_completion(messages)
            results = []
            for rank in raw_ranks["choices"][0]["message"]["content"].split(" > "):
                results.append(result_map[int(rank.strip("[]"))])
            return results    

        # self.session will be instantiated for ONNX based pairwise CE models
        else:
            self.logger.debug("Running pairwise ranking..")
            query_passage_pairs = [[query, passage["text"]] for passage in passages]

            input_text = self.tokenizer.encode_batch(query_passage_pairs)
            input_ids = np.array([e.ids for e in input_text])
            token_type_ids = np.array([e.type_ids for e in input_text])
            attention_mask = np.array([e.attention_mask for e in input_text])

            use_token_type_ids = "token_type_ids" in self.input_names

            onnx_input = {"input_ids": input_ids.astype(np.int64), "attention_mask": attention_mask.astype(np.int64)}
            if use_token_type_ids:
                onnx_input["token_type_ids"] = token_type_ids.astype(np.int64)

            outputs = self.session.run(None, onnx_input)

            logits = outputs[0]

            if logits.shape[1] == 1:
                scores = 1 / (1 + np.exp(-logits.flatten()))
            else:
                exp_logits = np.exp(logits)
                scores = exp_logits[:, 1] / np.sum(exp_logits, axis=1)

            for score, passage in zip(scores, passages):
                passage["score"] = score

            passages.sort(key=lambda x: x["score"], reverse=True)
            return passages
