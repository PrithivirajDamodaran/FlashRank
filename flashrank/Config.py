# Moved to HF - My cloud bucket and $ wont be abused anymore :-)
# THE HF Repo is UNDER CC-BY-SA LICENSE. USE APPROPRIATELY.
import os
from urllib.parse import urljoin

hf_endpoint = os.environ.get('HF_ENDPOINT', default='https://huggingface.co')
model_url = urljoin(hf_endpoint, 'prithivida/flashrank/resolve/main/{}.zip')
hf_model_url = urljoin(hf_endpoint, '{}/resolve/main/{}')
listwise_rankers = {'rank_zephyr_7b_v1_full'}
# Files a Hub repo must provide for a custom ONNX model to load.
required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
# Files that are useful when present but are legitimately absent from many repos.
optional_files = ['special_tokens_map.json']
# ONNX variants to probe for on the Hub, in order of preference. Two naming
# conventions appear in the wild: quantization variants from transformers.js /
# optimum (`model_quantized`, `model_q4`, ...) and Optimum optimization levels
# (`model_O1` through `model_O4`).
#
# The fp16 exports are tried last: several transformers.js fp16 graphs fail to
# initialize in ONNX Runtime with a SimplifiedLayerNormFusion error, so a
# plain-fp32 `model.onnx` is a safer fallback when a repo ships both.
onnx_file_candidates = [
    "onnx/model_quantized.onnx",
    "onnx/model_uint8.onnx",
    "onnx/model_int8.onnx",
    "onnx/model_q4.onnx",
    "onnx/model_bnb4.onnx",
    "onnx/model_O4.onnx",
    "onnx/model_O3.onnx",
    "onnx/model_O2.onnx",
    "onnx/model_O1.onnx",
    "onnx/model.onnx",
    "onnx/model_fp16.onnx",
    "onnx/model_q4f16.onnx",
]
default_cache_dir = "/tmp"
default_model = "ms-marco-TinyBERT-L-2-v2"
model_file_map = {
    "ms-marco-TinyBERT-L-2-v2": "flashrank-TinyBERT-L-2-v2.onnx",
    "ms-marco-MiniLM-L-12-v2": "flashrank-MiniLM-L-12-v2_Q.onnx",
    "ms-marco-MultiBERT-L-12": "flashrank-MultiBERT-L12_Q.onnx",
    "rank-T5-flan": "flashrank-rankt5_Q.onnx",
    "ce-esci-MiniLM-L12-v2": "flashrank-ce-esci-MiniLM-L12-v2_Q.onnx",
    "rank_zephyr_7b_v1_full": "rank_zephyr_7b_v1_full.Q4_K_M.gguf",
    "miniReranker_arabic_v1": "miniReranker_arabic_v1.onnx"
}

