import json
import hashlib
import pickle
from datetime import datetime
import fitz
import faiss

from sentence_transformers import SentenceTransformer
from core.config import (
    PDF_DIR,
    PDF_INDEX,
    PDF_META,
    PDF_REGISTRY,
    EMBED_MODEL
)

model = SentenceTransformer(EMBED_MODEL)


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry():
    if PDF_REGISTRY.exists():
        return json.loads(PDF_REGISTRY.read_text())
    return {"indexed_files": {}}


def save_registry(reg):
    PDF_REGISTRY.write_text(json.dumps(reg, indent=2))


def extract_text_chunks(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    words = text.split()

    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def incremental_index():

    registry = load_registry()
    indexed = registry["indexed_files"]

    new_files = []

    for pdf in PDF_DIR.glob("*.pdf"):
        h = file_hash(pdf)

        if pdf.name not in indexed or indexed[pdf.name]["hash"] != h:
            new_files.append((pdf, h))

    if not new_files:
        print("No new PDFs found")
        return

    if PDF_INDEX.exists():
        index = faiss.read_index(str(PDF_INDEX))
        meta = pickle.load(open(PDF_META, "rb"))
    else:
        index = None
        meta = []

    for pdf, h in new_files:

        print("Processing:", pdf.name)

        chunks = extract_text_chunks(pdf)
        vectors = model.encode(chunks)

        if index is None:
            index = faiss.IndexFlatL2(vectors.shape[1])

        index.add(vectors)

        # ⭐ IMPORTANT 
        for chunk in chunks:
            meta.append({
                "text": chunk,
                "source": pdf.name
            })

        registry["indexed_files"][pdf.name] = {
            "hash": h,
            "indexed_at": datetime.utcnow().isoformat()
        }

    faiss.write_index(index, str(PDF_INDEX))
    pickle.dump(meta, open(PDF_META, "wb"))
    save_registry(registry)

    print("PDF Index build complete")


if __name__ == "__main__":
    incremental_index()
