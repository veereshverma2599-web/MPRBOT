import os
import ollama

from services.retriever import retrieve_context


# =========================
# Ollama Configuration
# =========================
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")


def _normalize_context(context):
    """
    Ensure retrieved context is always a clean string.
    Prevents .strip() crashes.
    """

    if context is None:
        return ""

    # If retriever returns list
    if isinstance(context, list):
        return "\n".join(str(c) for c in context if c)

    # If retriever returns dict
    if isinstance(context, dict):
        return "\n".join(f"{k}: {v}" for k, v in context.items())

    return str(context)


def pdf_agent(question: str) -> str:
    """
    Answer a question strictly using retrieved PDF context.
    """

    # =========================
    # 1. Retrieve context
    # =========================
    try:
        raw_context = retrieve_context(question)
        context = _normalize_context(raw_context)
    except Exception as e:
        return f"Retriever error: {str(e)}"

    # =========================
    # 2. Empty context check
    # =========================
    if not context.strip():
        return "Not found in documents"

    # =========================
    # 3. Build RAG prompt
    # =========================
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

  
    #  Call Ollama
    
    try:
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

        return response["message"]["content"].strip()

    except Exception as e:
        return f"LLM error: {str(e)}"
