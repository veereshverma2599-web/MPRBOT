import time
import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------
# Path Setup
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INDEX_PATH = DATA_DIR / "pdf_index.faiss"
META_PATH = DATA_DIR / "pdf_meta.pkl"

# ---------------------------
# Load Resources ONCE
# ---------------------------
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

if not INDEX_PATH.exists():
    raise FileNotFoundError(f"FAISS index missing: {INDEX_PATH}")

if not META_PATH.exists():
    raise FileNotFoundError(f"Metadata missing: {META_PATH}")

INDEX = faiss.read_index(str(INDEX_PATH))

with open(META_PATH, "rb") as f:
    METADATA = pickle.load(f)

print("Retriever loaded ✅")

# ---------------------------
# Retrieve Context For RAG
# ---------------------------
def retrieve_context(query, top_k=5):

    if not query:
        return []

    start_time = time.time()

    query_embedding = MODEL.encode([query]).astype("float32")

    distances, indices = INDEX.search(query_embedding, top_k)

    results = []

    for rank, idx in enumerate(indices[0]):

        if idx < 0 or idx >= len(METADATA):
            continue

        case = METADATA[idx].copy()
        case["distance"] = float(distances[0][rank])

        results.append(case)

    print(f"Retrieval time: {(time.time()-start_time)*1000:.2f} ms")

    return results


# ---------------------------
# Format Context For LLM
# ---------------------------
def format_context(results):

    text = ""

    for r in results:
        text += f"\nCase: {r.get('case_id','N/A')}\n"
        text += f"Issue: {r.get('issue','')}\n"
        text += f"Resolution: {r.get('resolution','')}\n"

    return text


# ---------------------------
# Similar Case Search For UI
# ---------------------------
def find_similar_cases(query, model=None, index=None, metadata=None, top_k=5):

    """
    Used by UI to display similar historical cases
    """

    model = model or MODEL
    index = index or INDEX
    metadata = metadata or METADATA

    if not query:
        return []

    query_embedding = model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for rank, idx in enumerate(indices[0]):

        if idx < 0 or idx >= len(metadata):
            continue

        case = metadata[idx].copy()

        # Convert FAISS distance to confidence %
        confidence = max(0, 100 - float(distances[0][rank]))

        case["confidence"] = round(confidence, 2)

        results.append(case)

    return results
