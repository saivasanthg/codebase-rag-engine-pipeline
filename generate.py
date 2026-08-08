import os
import re
import time
import requests
from dotenv import load_dotenv
import ollama
from retrieve import format_context_for_llm, retrieve_code_context

load_dotenv()

SYSTEM_PROMPT = """You are an intelligent, polite, and articulate AI Software Engineering Partner.

Persona & Communication Style:
1. Conversational Etiquette: Speak with flawless grammar, warm professionalism, and respectful courtesy. Respond naturally to greetings, small talk, and general conversational prompts without being dry or robotic.
2. Topic Shift & Transition Awareness: Pay close attention to changes in conversation topics across turns. If the user pivots to a new topic, component, or file (or returns to an earlier discussed topic), explicitly acknowledge the transition (e.g., "Shifting from our earlier discussion on ingestion to model settings...") to maintain a cohesive, intelligent multi-turn narrative.
3. Natural Synthesis: When answering questions about a codebase, synthesize explanations naturally in clear prose. Explain *how* and *why* things work rather than simply dumping raw data or listing code snippets.
4. Code & Diagram Citations: When technical context is provided, reference specific file paths (`file_path`), line numbers, or image assets. Use concise code blocks (` ```lang ... ``` `) only when they illuminate your explanation.
5. Polite Context Boundaries: Rely on the provided codebase context for technical specifics. If the context does not contain enough information to answer a technical question, acknowledge it politely and explain what detail is missing.
6. Multi-Turn Memory & Continuity: Maintain smooth dialogue continuity with prior conversation history, referencing earlier context when relevant.
"""

REPHRASE_PROMPT = """Given the following conversation history and a follow-up user question, rephrase the follow-up question into a concise, standalone search query that contains all necessary technical context for a vector search engine.
Do NOT answer the question. Only output the rephrased standalone query. If the question is already self-contained or is a general greeting (like "hello", "thanks"), return it unchanged.

Conversation History:
{chat_history}

Follow-up Question: {query}
Standalone Query:"""

DEFAULT_CLOUDFLARE_MODEL = "@cf/meta/llama-3.1-8b-instruct"


def sanitize_history_for_llm(chat_history: list[dict], max_turns: int = 8) -> list[dict]:
    """Sanitizes past assistant turns to remove token bloat while preserving key dialogue context."""
    if not chat_history:
        return []

    sanitized = []
    recent_history = chat_history[-max_turns:]

    for msg in recent_history:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "assistant":
            clean_content = re.sub(r"```[\s\S]*?```", "[code block omitted for brevity]", content)
            if len(clean_content) > 300:
                clean_content = clean_content[:300] + "..."
            sanitized.append({"role": "assistant", "content": clean_content})
        else:
            sanitized.append({"role": "user", "content": content})

    return sanitized


def format_history_as_text(chat_history: list[dict], max_turns: int = 6) -> str:
    """Formats recent chat history as text for query rephrasing."""
    if not chat_history:
        return "No prior conversation."

    recent_turns = chat_history[-max_turns:]
    history_text = ""
    for msg in recent_turns:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        if role == "Assistant":
            content = re.sub(r"```[\s\S]*?```", "[code block]", content)
            if len(content) > 250:
                content = content[:250] + "..."
        history_text += f"{role}: {content}\n"
    return history_text.strip()


def generate_with_cloudflare(
    messages: list[dict],
    model_name: str = DEFAULT_CLOUDFLARE_MODEL,
    account_id: str = None,
    api_token: str = None,
    max_retries: int = 3,
) -> str:
    """Generates response using Cloudflare Workers AI REST API with automatic socket retry handling."""
    acc_id = account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID")
    token = api_token or os.getenv("CLOUDFLARE_API_TOKEN")

    if not acc_id or not token:
        raise ValueError(
            "Cloudflare Workers AI requires CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN."
        )

    if not model_name or not model_name.startswith("@cf/"):
        model_name = DEFAULT_CLOUDFLARE_MODEL

    url = f"https://api.cloudflare.com/client/v4/accounts/{acc_id}/ai/run/{model_name}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=35)
            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    return result["result"]["response"]
                else:
                    errors = result.get("errors", [])
                    raise RuntimeError(f"Cloudflare Workers AI returned success=False: {errors}")
            else:
                last_error = f"API error ({response.status_code}): {response.text}"
        except (requests.exceptions.RequestException, RuntimeError) as e:
            last_error = str(e)
            print(f"Network attempt {attempt}/{max_retries} failed ({e}). Retrying...")
            time.sleep(1.5)

    raise RuntimeError(f"Cloudflare Workers AI request failed after {max_retries} attempts: {last_error}")


def generate_with_ollama(model_name: str, messages: list[dict]) -> str:
    """Generates response using Ollama local API."""
    response = ollama.chat(
        model=model_name,
        messages=messages,
        options={"temperature": 0.3},
    )
    return response["message"]["content"]


def condense_question(
    query: str,
    chat_history: list[dict] = None,
    model_name: str = "phi4-mini",
    provider: str = "ollama",
    account_id: str = None,
    api_token: str = None,
) -> str:
    """Rewrites a follow-up question into a standalone search query if history exists."""
    if not chat_history:
        return query

    formatted_history = format_history_as_text(chat_history)
    prompt = REPHRASE_PROMPT.format(chat_history=formatted_history, query=query)

    try:
        if provider == "cloudflare":
            messages = [{"role": "user", "content": prompt}]
            rephrased = generate_with_cloudflare(
                messages=messages,
                model_name=model_name,
                account_id=account_id,
                api_token=api_token,
            ).strip()
            if rephrased:
                print(f"🔄 Rephrased standalone query: '{rephrased}'")
                return rephrased

        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        rephrased = response["message"]["content"].strip()
        if rephrased:
            print(f"🔄 Rephrased standalone query: '{rephrased}'")
            return rephrased
    except Exception as e:
        print(f"Warning: Query condensation failed ({e}). Using original query.")

    return query


def ask_codebase(
    query: str,
    chat_history: list[dict] = None,
    model_name: str = "phi4-mini",
    provider: str = "ollama",
    collection_name: str = "github_codebase",
    top_k_retrieve: int = 15,
    top_k_rerank: int = 3,
    account_id: str = None,
    api_token: str = None,
):
    """Main RAG pipeline: Query condensation -> Multi-Query & Self-Query Retrieval -> CrossEncoder -> LLM."""
    standalone_query = condense_question(
        query=query,
        chat_history=chat_history,
        model_name=model_name,
        provider=provider,
        account_id=account_id,
        api_token=api_token,
    )

    print(f"\n🔍 Searching vector store for query: '{standalone_query}'...")
    top_snippets = retrieve_code_context(
        query=standalone_query,
        collection_name=collection_name,
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
        provider=provider,
        model_name=model_name,
        account_id=account_id,
        api_token=api_token,
    )

    if not top_snippets:
        formatted_context = "Note: No specific codebase snippets were retrieved for this conversational turn."
    else:
        formatted_context = format_context_for_llm(top_snippets)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        sanitized_history = sanitize_history_for_llm(chat_history, max_turns=8)
        for turn in sanitized_history:
            messages.append({"role": turn["role"], "content": turn["content"]})

    user_prompt = f"{formatted_context}\n---\nUser Message: {query}"
    messages.append({"role": "user", "content": user_prompt})

    print(f"🤖 Generating response using provider='{provider}' model='{model_name}'...\n")

    if provider == "cloudflare":
        answer = generate_with_cloudflare(
            messages=messages,
            model_name=model_name,
            account_id=account_id,
            api_token=api_token,
        )
    else:
        answer = generate_with_ollama(model_name=model_name, messages=messages)

    return answer, top_snippets, standalone_query


if __name__ == "__main__":
    history = [
        {"role": "user", "content": "How does ingest.py work?"},
        {"role": "assistant", "content": "It clones the repo and splits code files into chunks stored in Qdrant."},
        {"role": "user", "content": "Now tell me about Cloudflare Workers AI in generate.py"},
    ]
    followup_query = "How does it handle API tokens?"

    answer, snippets, search_q = ask_codebase(
        query=followup_query,
        chat_history=history,
        provider="ollama",
        model_name="phi4-mini",
    )

    print("=" * 60)
    print(f"🔍 Search Query Used: {search_q}")
    print("=" * 60)
    print("💡 ANSWER:")
    print(answer)