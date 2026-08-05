import os
import requests
import json
import ollama
from retrieve import format_context_for_llm, retrieve_code_context
import config

SYSTEM_PROMPT = """You are an expert Software Engineering AI assistant.
Your task is to answer user queries about a code repository using ONLY the provided code snippets below.

Rules:
1. Reference specific file paths (`file_path`) and function/class names when answering.
2. If the context does not contain enough information to answer, explicitly state that.
3. Keep code explanations clear, concise, and accurate based on the context.
"""

def generate_with_cloudflare(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Generates LLM response using Cloudflare Workers AI API."""
    account_id = config.CLOUDFLARE_ACCOUNT_ID
    api_token = config.CLOUDFLARE_API_TOKEN
    model_name = config.CLOUDFLARE_MODEL  # Defaults to @cf/meta/llama-3.1-8b-instruct

    if not account_id or not api_token:
        return (
            "⚠️ Cloudflare Credentials Missing! Please set `CLOUDFLARE_ACCOUNT_ID` and "
            "`CLOUDFLARE_API_TOKEN` in `.env` or environment variables."
        )

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        if res.status_code == 200:
            data = res.json()
            if data.get("success"):
                result = data.get("result", {})
                if "response" in result:
                    return result["response"]
                elif "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    return json.dumps(result)
            else:
                return f"❌ Cloudflare Workers AI Error: {data.get('errors', [])}"
        else:
            return f"❌ HTTP Error {res.status_code} from Cloudflare API: {res.text}"
    except Exception as e:
        return f"❌ Cloudflare Workers AI Exception: {str(e)}"

def generate_with_ollama(prompt: str, model_name: str = config.OLLAMA_MODEL, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Generates LLM response using local Ollama instance."""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.2},
        )
        return response["message"]["content"]
    except Exception as e:
        return f"❌ Ollama Error: {str(e)}"

def ask_codebase(
    query: str,
    provider: str = config.DEFAULT_PROVIDER,
    top_k_retrieve: int = 15,
    top_k_rerank: int = 3
) -> str:
    print(f"\n🔍 Searching repository for: '{query}'...")

    # 1. Retrieve & Rerank snippets
    top_snippets = retrieve_code_context(
        query=query, top_k_retrieve=top_k_retrieve, top_k_rerank=top_k_rerank
    )

    if not top_snippets:
        return "No relevant code snippets were found in the codebase repository."

    # 2. Format context string
    formatted_context = format_context_for_llm(top_snippets)

    # 3. Construct message payload
    user_prompt = f"""{formatted_context}

---

User Question: {query}"""

    # 4. Generate Answer using chosen provider
    if provider.lower() in ["cloudflare", "cf", "cloudflare_workers"]:
        print(f"🤖 Generating response using Cloudflare Workers AI ({config.CLOUDFLARE_MODEL})...\n")
        return generate_with_cloudflare(user_prompt)
    else:
        print(f"🤖 Generating response using Ollama ({config.OLLAMA_MODEL})...\n")
        return generate_with_ollama(user_prompt)

if __name__ == "__main__":
    test_query = "How is it generating the resume?"
    
    # Example using Cloudflare Workers AI API
    print("Testing with Cloudflare Workers AI:")
    answer_cf = ask_codebase(test_query, provider="cloudflare")
    print("=" * 60)
    print("💡 CLOUDFLARE ANSWER:")
    print("=" * 60)
    print(answer_cf)