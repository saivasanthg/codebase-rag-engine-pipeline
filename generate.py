import ollama
from retrieve import format_context_for_llm, retrieve_code_context

# System prompt forcing the model to rely only on retrieved repo context
SYSTEM_PROMPT = """You are an expert Software Engineering AI assistant.
Your task is to answer user queries about a code repository using ONLY the provided code snippets below.

Rules:
1. Reference specific file paths (`file_path`) and function/class names when answering.
2. If the context does not contain enough information to answer, explicitly state that.
3. Keep code explanations clear, concise, and accurate based on the context.
"""


def ask_codebase(
    query: str, model_name: str = "phi4-mini", top_k_retrieve=15, top_k_rerank=3
):
    print(f"\n🔍 Searching repository for: '{query}'...")

    # 1. Retrieve & Rerank snippets using your Phase 3 pipeline
    top_snippets = retrieve_code_context(
        query=query, top_k_retrieve=top_k_retrieve, top_k_rerank=top_k_rerank
    )

    if not top_snippets:
        return "No relevant code snippets were found in the codebase repository."

    # 2. Format context string
    formatted_context = format_context_for_llm(top_snippets)

    # 3. Construct message payload
    user_prompt = f"""

{formatted_context}

---

User Question: {query}"""

    print(f"🤖 Generating response using Ollama ({model_name})...\n")

    # 4. Call Ollama locally
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": 0.2  # Low temperature for precise code explanations
        },
    )

    return response["message"]["content"]


if __name__ == "__main__":
    # Test Question
    test_query = "How is it generating the resume?"

    answer = ask_codebase(test_query)

    print("=" * 60)
    print("💡 ANSWER:")
    print("=" * 60)
    print(answer)