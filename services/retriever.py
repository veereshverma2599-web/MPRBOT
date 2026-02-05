import time
import pickle
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------
# Path Setup
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INDEX_PATH = DATA_DIR / "mpr_index.faiss"
META_PATH = DATA_DIR / "mpr_meta.pkl"

# ---------------------------
# Load Resources ONCE
# ---------------------------
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
INDEX = faiss.read_index(str(INDEX_PATH))

with open(META_PATH, "rb") as f:
    METADATA = pickle.load(f)

print("Retriever loaded ✅")


# ---------------------------
# Retrieve Similar Cases
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
