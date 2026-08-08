import os
import re
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

load_dotenv()

_embedding_model = None
_reranker_model = None
_qdrant_client = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        print(f"Loading embedding model: {model_name}...")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        model_name = os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        print(f"Loading reranker model: {model_name}...")
        _reranker_model = CrossEncoder(model_name)
    return _reranker_model


def connect_to_qdrant() -> QdrantClient:
    """Connects to Qdrant using a singleton instance to prevent embedded file lock conflicts."""
    global _qdrant_client
    if _qdrant_client is None:
        qdrant_url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url:
            print(f"Connecting to remote Qdrant server: {qdrant_url}")
            _qdrant_client = QdrantClient(url=qdrant_url, api_key=api_key)
        else:
            qdrant_dir = os.path.abspath("./qdrant_db")
            print(f"Connecting to local embedded Qdrant database at: {qdrant_dir}")
            _qdrant_client = QdrantClient(path=qdrant_dir)
    return _qdrant_client


def extract_file_filter_from_query(query: str) -> str:
    """LangChain Self-Querying: Extracts file path or extension filters if mentioned in user query."""
    match = re.search(r"\b(?:in|inside|from|file)\s+([a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]+)\b", query, re.IGNORECASE)
    if match:
        file_target = match.group(1).replace("\\", "/")
        print(f"Self-Query Filter Detected: Target File = '{file_target}'")
        return file_target
    return None


def generate_query_variations(query: str, provider: str = "ollama", model_name: str = "phi4-mini", account_id: str = None, api_token: str = None) -> list[str]:
    """LangChain Multi-Query Expansion: Generates 2 technical variations of a user query for multi-perspective search."""
    if len(query.strip().split()) <= 2:
        return [query]

    prompt = f"Generate 2 concise, alternative technical search queries for a code vector search engine based on this question: '{query}'. Output ONLY the 2 queries separated by newlines."
    
    variations = [query]
    try:
        if provider == "cloudflare":
            acc_id = account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID")
            token = api_token or os.getenv("CLOUDFLARE_API_TOKEN")
            if acc_id and token:
                url = f"https://api.cloudflare.com/client/v4/accounts/{acc_id}/ai/run/@cf/meta/llama-3.1-8b-instruct"
                headers = {"Authorization": f"Bearer {token}"}
                payload = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
                res = requests.post(url, headers=headers, json=payload, timeout=10)
                if res.status_code == 200 and res.json().get("success"):
                    lines = res.json()["result"]["response"].strip().split("\n")
                    for line in lines:
                        cleaned = line.strip().strip("1234567890.- ")
                        if cleaned and cleaned not in variations:
                            variations.append(cleaned)
        else:
            res = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.2})
            lines = res["message"]["content"].strip().split("\n")
            for line in lines:
                cleaned = line.strip().strip("1234567890.- ")
                if cleaned and cleaned not in variations:
                    variations.append(cleaned)
    except Exception as e:
        print(f"Multi-query expansion skipped ({e})")

    print(f"Multi-Query Variations ({len(variations)}): {variations}")
    return variations[:3]


def retrieve_code_context(
    query: str,
    collection_name: str = "github_codebase",
    top_k_retrieve: int = 20,
    top_k_rerank: int = 3,
    provider: str = "ollama",
    model_name: str = "phi4-mini",
    account_id: str = None,
    api_token: str = None,
):
    """Retrieves relevant code/image chunks using LangChain Multi-Query + Self-Query Filtering + Qdrant + CrossEncoder."""
    client = connect_to_qdrant()

    if not client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' does not exist in Qdrant.")
        return []

    embedding_model = get_embedding_model()
    reranker = get_reranker_model()

    file_filter_target = extract_file_filter_from_query(query)
    qdrant_filter = None
    if file_filter_target:
        qdrant_filter = Filter(
            must=[FieldCondition(key="file_path", match=MatchValue(value=file_filter_target))]
        )

    query_variations = generate_query_variations(
        query, provider=provider, model_name=model_name, account_id=account_id, api_token=api_token
    )

    all_hits = {}
    for q_var in query_variations:
        query_vector = embedding_model.encode(q_var).tolist()
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="text_vector",
            query_filter=qdrant_filter,
            limit=top_k_retrieve,
            with_payload=True,
        ).points

        for hit in results:
            chunk_id = hit.payload.get("chunk_id", str(hit.id))
            if chunk_id not in all_hits:
                all_hits[chunk_id] = hit

    if not all_hits:
        if qdrant_filter:
            print(f"No hits with filter '{file_filter_target}'. Falling back to global search...")
            return retrieve_code_context(
                query=query,
                collection_name=collection_name,
                top_k_retrieve=top_k_retrieve,
                top_k_rerank=top_k_rerank,
                provider=provider,
                model_name=model_name,
                account_id=account_id,
                api_token=api_token,
            )
        return []

    unique_hits = list(all_hits.values())

    candidate_documents = [hit.payload.get("code", "") for hit in unique_hits]
    query_doc_pairs = [[query, doc] for doc in candidate_documents]

    rerank_scores = reranker.predict(query_doc_pairs)

    scored_candidates = []
    for idx, hit in enumerate(unique_hits):
        scored_candidates.append(
            {
                "score": float(rerank_scores[idx]),
                "file_path": hit.payload.get("file_path", "unknown"),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "start_line": hit.payload.get("start_line", 1),
                "end_line": hit.payload.get("end_line", 1),
                "code": hit.payload.get("code", ""),
                "is_image": hit.payload.get("is_image", False),
                "image_path": hit.payload.get("image_path", ""),
            }
        )

    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    return scored_candidates[:top_k_rerank]


def format_context_for_llm(results):
    """Combines retrieved code chunks and image descriptions into a structured prompt context block."""
    if not results:
        return "No relevant code or image snippets were retrieved from the repository."

    context_str = "Below is relevant source code and repository image descriptions retrieved from the repository:\n\n"
    for idx, res in enumerate(results, start=1):
        if res.get("is_image"):
            context_str += f"--- Snippet {idx}: [REPOSITORY IMAGE FILE] {res['file_path']} (Score: {res['score']:.4f}) ---\n"
        else:
            line_info = f" (Lines {res['start_line']}-{res['end_line']})" if res.get('start_line') else ""
            context_str += f"--- Snippet {idx}: {res['file_path']}{line_info} (Score: {res['score']:.4f}) ---\n"
        context_str += f"{res['code']}\n\n"
    return context_str


if __name__ == "__main__":
    user_query = "Where is Qdrant configured in ingest.py?"
    print(f"\nSearching for: '{user_query}'...\n")
    top_snippets = retrieve_code_context(user_query, top_k_retrieve=15, top_k_rerank=3)

    for snippet in top_snippets:
        print(
            f"Score: {snippet['score']:.4f} | File: {snippet['file_path']} | Is Image: {snippet.get('is_image')}"
        )
        print(f"Code Preview:\n{snippet['code'][:200]}...\n")
        print("=" * 60)