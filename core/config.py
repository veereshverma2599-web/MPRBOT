import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"

PDF_INDEX = DATA_DIR / "pdf_index.faiss"
PDF_META = DATA_DIR / "pdf_meta.pkl"
PDF_REGISTRY = DATA_DIR / "index_registry.json"

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")

TOP_K = int(os.getenv("TOP_K", 3))
