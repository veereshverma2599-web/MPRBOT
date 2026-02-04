import os
import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime

import faiss
from sentence_transformers import SentenceTransformer

# =========================
# PATH SETUP
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = BASE_DIR / "pdfs"
DATA_DIR = BASE_DIR / "data"

INDEX_PATH = DATA_DIR / "pdf_index.faiss"
META_PATH = DATA_DIR / "pdf_meta.pkl"
REGISTRY_PATH = DATA_DIR / "index_registry.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# HELPERS
# =========================
def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def load_registry():
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"indexed_files": {}}

def save_registry(reg):
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))

# =========================
# PDF DISCOVERY
# =========================
def get_new_pdfs():
    registry = load_registry()
    indexed = registry["indexed_files"]

    new_files = []
    for pdf in PDF_DIR.glob("*.pdf"):
        h = file_hash(pdf)
        if pdf.name not in indexed or indexed[pdf.name]["hash"] != h:
            new_files.append((pdf, h))

    return new_files, registry

# =========================
# MAIN INCREMENTAL INDEXER
# =========================
def incremental_index():
    new_files, registry = get_new_pdfs()

    if not new_files:
        print("No new PDFs found")
        return

    # Load or create FAISS
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        meta = pickle.load(open(META_PATH, "rb"))
    else:
        index = None
        meta = []

    for pdf, h in new_files:
        texts = extract_text_chunks(pdf)   
        vectors = model.encode(texts)

        if index is None:
            index = faiss.IndexFlatL2(vectors.shape[1])

        index.add(vectors)
        meta.extend([{"source": pdf.name}] * len(texts))

        registry["indexed_files"][pdf.name] = {
            "hash": h,
            "indexed_at": datetime.utcnow().isoformat()
        }

        print(f"Indexed: {pdf.name}")

    faiss.write_index(index, str(INDEX_PATH))
    pickle.dump(meta, open(META_PATH, "wb"))
    save_registry(registry)

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    incremental_index()
