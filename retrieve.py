from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import config

# Load First-Stage Embedding Model & Second-Stage Cross-Encoder Reranker
print("Loading embedding and reranker models...")
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
reranker = CrossEncoder(config.RERANKER_MODEL_NAME)

def connect_to_qdrant(qdrant_url: str = config.QDRANT_URL):
    return QdrantClient(url=qdrant_url)

def retrieve_code_context(query: str, collection_name=config.DEFAULT_COLLECTION_NAME, top_k_retrieve=20, top_k_rerank=3):
    client = connect_to_qdrant()

    if not client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' does not exist in Qdrant.")
        return []

    query_vector = embedding_model.encode(query).tolist()

    search_results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k_retrieve,
        with_payload=True
    ).points

    if not search_results:
        print("No matches found in Qdrant.")
        return []

    # Prepare document text pairs for Cross-Encoder scoring
    candidate_documents = [hit.payload["code"] for hit in search_results]
    query_doc_pairs = [[query, doc] for doc in candidate_documents]

    # Step 3: Rerank candidates using Cross-Encoder
    rerank_scores = reranker.predict(query_doc_pairs)

    # Attach rerank scores and sort candidates
    scored_candidates = []
    for idx, hit in enumerate(search_results):
        scored_candidates.append({
            "score": float(rerank_scores[idx]),
            "file_path": hit.payload["file_path"],
            "chunk_index": hit.payload["chunk_index"],
            "code": hit.payload["code"]
        })

    # Sort descending by reranker score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Return top N reranked results
    return scored_candidates[:top_k_rerank]

def format_context_for_llm(results):
    """Combines retrieved code chunks into a structured prompt context block."""
    context_str = "Below is relevant source code retrieved from the repository:\n\n"
    for idx, res in enumerate(results, start=1):
        context_str += f"--- Snippet {idx} (File: {res['file_path']}) ---\n"
        context_str += f"{res['code']}\n\n"
    return context_str

if __name__ == "__main__":
    user_query = "What are the packages used in this project?"
    print(f"\n🔍 Searching for: '{user_query}'...\n")
    top_snippets = retrieve_code_context(user_query, top_k_retrieve=15, top_k_rerank=3)
    
    for snippet in top_snippets:
        print(f"🎯 Score: {snippet['score']:.4f} | File: {snippet['file_path']}")
        print(f"Code Preview:\n{snippet['code'][:200]}...\n")
        print("=" * 60)