import pickle
import faiss
from sentence_transformers import SentenceTransformer

from core.config import PDF_INDEX, PDF_META, EMBED_MODEL, TOP_K

MODEL = SentenceTransformer(EMBED_MODEL)

INDEX = faiss.read_index(str(PDF_INDEX))

with open(PDF_META, "rb") as f:
    METADATA = pickle.load(f)


def retrieve_context(query):

    if not query.strip():
        return ""

    query_vec = MODEL.encode([query]).astype("float32")
    _, indices = INDEX.search(query_vec, TOP_K)

    chunks = []

    for i in indices[0]:
        if i < len(METADATA):
            text = METADATA[i].get("text", "")
            if text:
                chunks.append(text)

    return "\n".join(chunks)
