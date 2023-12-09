import os
import json
import zipfile
import requests
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from tokenizers import AddedToken, Tokenizer

from . import config

class Ranker:
    """
    Setup a ranker instance
    :param model_id: Model to use for reranking

    >>> from flashrank import Ranker
    >>> ranker = Ranker(model_id=...)
    >>> ranker.rerank(
    ...     query="<query>", passages=["<passage1>", "<passage2>", ...]    
    ... )
    """
    def __init__(self, model_id: str = config.DEFAULT_MODEL):

        self.model_id = model_id

        model_file = config.model_file_map.get(model_id)
        if model_file is None:
            raise LookupError(f"{model_id!r} model is not available.")
        
        self.model_dir = config.DEFAULT_CACHE_DIR / model_id
        if not self.model_dir.exists():
            self.__download_model()
        
        self.session = ort.InferenceSession(self.model_dir / model_file)
        self.tokenizer = self.load_tokenizer()

    def __download_model(self):
        """Download the Model"""
        local_zip_file = str(config.DEFAULT_CACHE_DIR / f"{self.model_id}.zip")
        model_url = config.get_model_url(self.model_id)

        with requests.get(model_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            progress_bar = tqdm(
                desc=self.model_id, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,
            )
            with open(local_zip_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    progress_bar.update(f.write(chunk))
            
            progress_bar.desc = local_zip_file
            progress_bar.close()

        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(config.DEFAULT_CACHE_DIR)

        os.remove(local_zip_file)

    def __json_file_handler(self, filename: str, read: bool = True) -> dict[str, Any] | Path:
        """Json file handler. If read is true, returns loaded object else path is returned"""
        path = self.model_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"{filename!r} file missing from {self.model_dir!r}")
        if read:
            return json.loads(path.read_bytes())
        return path

    def load_tokenizer(self, max_length: int = 512) -> Tokenizer:
        """Load tokenizer"""
        config = self.__json_file_handler("config.json")
        tokenizer_config = self.__json_file_handler("tokenizer_config.json")
        tokens_map = self.__json_file_handler("special_tokens_map.json")

        tokenizer: Tokenizer = Tokenizer.from_file(str(
            self.__json_file_handler("tokenizer.json", read=False)
        ))
        tokenizer.enable_truncation(max_length=min(tokenizer_config["model_max_length"], max_length))
        tokenizer.enable_padding(pad_id=config["pad_token_id"], pad_token=tokenizer_config["pad_token"])

        for token in tokens_map.values():
            if isinstance(token, str):
                tokenizer.add_special_tokens([token])
            elif isinstance(token, dict):
                tokenizer.add_special_tokens([AddedToken(**token)])

        return tokenizer
    
    def __create_attr_array(self, tokenized, attr: str):
        """Create array of tokenized attribute values."""
        return np.array([getattr(_, attr) for _ in tokenized], dtype=np.int64)

    def rerank(self, query: str, passages: Iterable[str]) -> dict[str, float | str]:
        """Rerank passages based on query."""
        tokenized = self.tokenizer.encode_batch(
            [(query, passage) for passage in passages]
        )
        onnx_input = {
            "input_ids": self.__create_attr_array(tokenized, 'ids'),
            "token_type_ids": self.__create_attr_array(tokenized, 'type_ids'),
            "attention_mask": self.__create_attr_array(tokenized, 'attention_mask'),
        }

        output = self.session.run(None, onnx_input)[0]
        scores = output[:, 1] if output.shape[1] > 1 else output.flatten()
        
        scores = list(1 / (1 + np.exp(-scores)))
        combined_passages = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)

        return [{'score': s, "passage": p} for s, p in combined_passages]

