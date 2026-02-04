import os
import ollama
from services.retriever import retrieve_context

# =========================
# Ollama Configuration
# =========================
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")

def pdf_agent(question: str) -> str:
    """
    Answer a question strictly using retrieved PDF context.
    If answer is not found, respond clearly.
    """

    # 1. Retrieve context from FAISS / PDFs
    context = retrieve_context(question)

    if not context or not context.strip():
        return "Not found in documents"

    # 2. Build strict RAG prompt
    prompt = f"""
You are a document-based assistant.

RULES:
- Answer ONLY using the provided context
- Do NOT use outside knowledge
- If answer is not in the context, reply exactly:
  "Not found in documents"

Context:
{context}

Question:
{question}

Answer:
""".strip()

    # 3. Call Ollama LLM
    response = ollama.chat(
        model=OLLAMA_LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": 0.2,
            "top_p": 0.9
        }
    )

    # 4. Return clean answer
    return response["message"]["content"].strip()
