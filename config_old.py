import os
from pathlib import Path

# =========================
# Base Paths
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"

# =========================
# FAISS Files
# =========================
PDF_INDEX = DATA_DIR / "pdf_index.faiss"
PDF_META = DATA_DIR / "pdf_meta.pkl"
PDF_REGISTRY = DATA_DIR / "index_registry.json"

# =========================
# Embedding Model
# =========================
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "all-MiniLM-L6-v2"
)

# =========================
# Ollama Model
# =========================
OLLAMA_MODEL = os.getenv(
    "OLLAMA_LLM_MODEL",
    "llama3.1:8b"
)

# =========================
# Retrieval Settings
# =========================
TOP_K = int(os.getenv("TOP_K", 5))
