model_url = 'https://storage.googleapis.com/flashrank/{}.zip'
default_cache_dir = "/tmp"
default_model = "ms-marco-TinyBERT-L-2-v2"
model_file_map = {
    "ms-marco-TinyBERT-L-2-v2": "flashrank-TinyBERT-L-2-v2.onnx",
    "ms-marco-MiniLM-L-12-v2": "flashrank-MiniLM-L-12-v2_Q.onnx",
    "ms-marco-MultiBERT-L-12": "flashrank-MultiBERT-L12_Q.onnx",
    "rank-T5-flan": "flashrank-rankt5_Q.onnx"
}