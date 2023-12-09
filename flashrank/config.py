import os
from pathlib import Path

DEFAULT_CACHE_DIR = Path(os.getenv(
    "FLASHRANK_CACHE", 
    default=Path("~").expanduser() / ".cache" / "flashrank"
))
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "ms-marco-TinyBERT-L-2-v2"
"""Default Model to use"""

model_file_map = {
    "ms-marco-TinyBERT-L-2-v2": "flashrank-TinyBERT-L-2-v2.onnx",
    "ms-marco-MiniLM-L-12-v2": "flashrank-MiniLM-L-12-v2_Q.onnx",
    "ms-marco-MultiBERT-L-12": "flashrank-MultiBERT-L12_Q.onnx"
}
"""Model to file mappings"""

def get_model_url(model_id: str):
    """Get formatted model url"""
    return f'https://storage.googleapis.com/flashrank/{model_id}.zip'
