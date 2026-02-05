import ollama

from core.config import OLLAMA_MODEL
from services.retriever import retrieve_context


def pdf_agent(question: str) -> str:

    context = retrieve_context(question)

    if not context.strip():
        return "Not found in documents"

    prompt = f"""
You are a document-based assistant.

RULES:
- Answer ONLY using context
- If answer missing say:
  "Not found in documents"

Context:
{context}

Question:
{question}

Answer:
""".strip()

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"].strip()
