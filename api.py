import os
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from ingest import clone_repository, process_repository, store_in_qdrant, get_collection_name_for_repo
from generate import ask_codebase

load_dotenv()

app = FastAPI(
    title="GitHub Repo LLM - Multimodal RAG API",
    description="Decoupled production REST API for repository ingestion, vector retrieval, and multimodal LLM generation.",
    version="1.0.0",
)

# Enable CORS for cross-origin frontend clients (Streamlit, React, Next.js, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Schemas ---
class IngestRequest(BaseModel):
    repo_url: str = Field(..., example="https://github.com/maheshpaulj/ResumeItNow")


class IngestResponse(BaseModel):
    status: str
    message: str
    repo_url: str
    collection_name: str
    chunks_indexed: int


class ChatRequest(BaseModel):
    query: str = Field(..., example="What is this repository about?")
    chat_history: Optional[List[Dict[str, str]]] = Field(default=[], example=[])
    provider: Optional[str] = Field(default="ollama", example="cloudflare")
    model_name: Optional[str] = Field(default="phi4-mini", example="@cf/meta/llama-3.1-8b-instruct")
    collection_name: Optional[str] = Field(default="github_codebase")
    top_k_retrieve: Optional[int] = Field(default=15)
    top_k_rerank: Optional[int] = Field(default=3)
    account_id: Optional[str] = None
    api_token: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    snippets: List[Dict]
    standalone_query: str


# --- REST API Endpoints ---
@app.get("/")
def read_root():
    return {
        "title": "GitHub Repo LLM - Multimodal RAG API",
        "docs": "http://localhost:8000/docs",
        "status": "online",
    }


@app.get("/api/health")
def check_health():
    return {
        "status": "healthy",
        "qdrant_storage": "local_embedded",
        "api_version": "1.0.0",
    }


@app.post("/api/ingest", response_model=IngestResponse)
def ingest_repository(payload: IngestRequest):
    """Clones a GitHub repository, extracts code & images, generates embeddings, and indexes into Qdrant."""
    repo_url = payload.repo_url.strip()
    if not repo_url:
        raise HTTPException(status_code=400, detail="Repository URL cannot be empty.")

    try:
        coll_name = get_collection_name_for_repo(repo_url)
        repo_dir = clone_repository(repo_url)
        docs = process_repository(repo_dir)
        store_in_qdrant(docs, collection_name=coll_name)

        return IngestResponse(
            status="success",
            message=f"Successfully cloned and indexed {len(docs)} document chunks.",
            repo_url=repo_url,
            collection_name=coll_name,
            chunks_indexed=len(docs),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
def chat_with_codebase(payload: ChatRequest):
    """Executes Multimodal RAG pipeline (Query expansion -> Qdrant retrieval -> CrossEncoder -> LLM answer)."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        answer, snippets, standalone_query = ask_codebase(
            query=payload.query,
            chat_history=payload.chat_history,
            model_name=payload.model_name,
            provider=payload.provider,
            collection_name=payload.collection_name,
            top_k_retrieve=payload.top_k_retrieve,
            top_k_rerank=payload.top_k_rerank,
            account_id=payload.account_id,
            api_token=payload.api_token,
        )

        return ChatResponse(
            answer=answer,
            snippets=snippets,
            standalone_query=standalone_query,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
