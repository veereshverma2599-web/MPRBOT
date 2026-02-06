import ollama
from core.config import OLLAMA_MODEL
from services.retriever import retrieve_context


def pdf_agent(question: str) -> str:

    print("STEP 1: Agent started")

    context = retrieve_context(question)

    # 🔥 Limit context size
    MAX_CONTEXT = 500
    context = context[:MAX_CONTEXT]

    print("STEP 2: Context retrieved")
    print("Context length:", len(context))

    if not context.strip():
        return "Not found in documents"

    prompt = f"""
Use the context to answer the question.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": 250,
                "temperature": 0.2
            }
        )

        print("STEP 3: LLM responded")

        return response["message"]["content"].strip()

    except Exception as e:
        return f"LLM Error: {str(e)}"
