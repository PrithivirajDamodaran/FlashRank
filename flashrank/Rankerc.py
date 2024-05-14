import json
from pathlib import Path
from tokenizers import AddedToken, Tokenizer
import onnxruntime as ort
import numpy as np
import os
import zipfile
import requests
from tqdm import tqdm
from flashrank.Config import default_model, default_cache_dir, model_url, model_file_map
import collections

class RerankRequest:

    def __init__(self, query=None, passages=None):
        self.query = query
        self.passages = passages if passages is not None else []

class Ranker:

    def __init__(self, 
                 model_name = default_model, 
                 cache_dir= default_cache_dir,
                 max_length = 512):

        self.cache_dir = Path(cache_dir)
        
        if not self.cache_dir.exists():
            print(f"Cache directory {self.cache_dir} not found. Creating it..")
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_dir = self.cache_dir / model_name
        
        if not self.model_dir.exists():
            print(f"Downloading {model_name}...")
            self._download_model_files(model_name)
            
        model_file = model_file_map[model_name]
        
        
        self.session = ort.InferenceSession(self.cache_dir / model_name / model_file)
        self.tokenizer = self._get_tokenizer(max_length)

    def _download_model_files(self, model_name):
        
        # The local file path to which the file should be downloaded
        local_zip_file = self.cache_dir / f"{model_name}.zip"

        formatted_model_url = model_url.format(model_name)

        with requests.get(formatted_model_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_zip_file, 'wb') as f, tqdm(
                    desc=local_zip_file.name,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

        # Extract the zip file
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.cache_dir)

        # Optionally, remove the zip file after extraction
        os.remove(local_zip_file)

    def _load_vocab(self, vocab_file):
    
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab
        

    def _get_tokenizer(self, max_length = 512):
      
      config_path = self.model_dir / "config.json"
      if not config_path.exists():
          raise FileNotFoundError(f"config.json missing in {self.model_dir}")

      tokenizer_path = self.model_dir / "tokenizer.json"
      if not tokenizer_path.exists():
          raise FileNotFoundError(f"tokenizer.json missingin  {self.model_dir}")

      tokenizer_config_path = self.model_dir / "tokenizer_config.json"
      if not tokenizer_config_path.exists():
          raise FileNotFoundError(f"tokenizer_config.json missing in  {self.model_dir}")

      tokens_map_path = self.model_dir / "special_tokens_map.json"
      if not tokens_map_path.exists():
          raise FileNotFoundError(f"special_tokens_map.json missing in  {self.model_dir}")

      config = json.load(open(str(config_path)))
      tokenizer_config = json.load(open(str(tokenizer_config_path)))
      tokens_map = json.load(open(str(tokens_map_path)))

      tokenizer = Tokenizer.from_file(str(tokenizer_path))
      tokenizer.enable_truncation(max_length=min(tokenizer_config["model_max_length"], max_length))
      tokenizer.enable_padding(pad_id=config["pad_token_id"], pad_token=tokenizer_config["pad_token"])

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
    
    
    def rerank(self, request):
        query = request.query
        passages = request.passages

        query_passage_pairs = [[query, passage["text"]] for passage in passages]

        input_text = self.tokenizer.encode_batch(query_passage_pairs)
        input_ids = np.array([e.ids for e in input_text])
        token_type_ids = np.array([e.type_ids for e in input_text])
        attention_mask = np.array([e.attention_mask for e in input_text])
        
        use_token_type_ids = token_type_ids is not None and not np.all(token_type_ids == 0)

        if use_token_type_ids:
            onnx_input = {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64),
                "token_type_ids": np.array(token_type_ids, dtype=np.int64),
            }
        else:
            onnx_input = {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64)
            }


        input_data = {k: v for k, v in onnx_input.items()}

        outputs = self.session.run(None, input_data)

        if outputs[0].shape[1] > 1:
            scores = outputs[0][:, 1]
        else:
            scores = outputs[0].flatten()

        scores = list(1 / (1 + np.exp(-scores)))  

        # Combine scores with passages, including metadata
        for score, passage in zip(scores, passages):
            passage["score"] = score

        # Sort passages based on scores
        passages.sort(key=lambda x: x["score"], reverse=True)

        return passages
    


