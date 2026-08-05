# 🚀 Codebase RAG Engine Pipeline

An intelligent Retrieval-Augmented Generation (RAG) system designed to analyze, index, search, and answer complex technical questions about any GitHub code repository.

Equipped with **AST language-aware code chunking**, **BGE dense vector retrieval**, **Cross-Encoder reranking**, **Local Ollama generation**, and **Cloudflare Workers AI API integration**.

---

## 📌 Project Architecture

```mermaid
flowchart TD
    subgraph Ingestion_Pipeline ["1. Ingestion & Indexing Pipeline"]
        A[Git Repository URL] --> B[Clone Repo & Filter Files]
        B --> C[AST Language-Aware Splitter\n.py, .js, .ts, .cpp, .java, .go, .md]
        C --> D[BGE Embedding Generator\nBAAI/bge-small-en-v1.5]
        D --> E[(Qdrant Vector DB)]
    end

    subgraph Query_Pipeline ["2. Retrieval, Reranking & LLM Generation"]
        F[User Query] --> G[Dense Vector Search\nTop-15 Retrieval]
        E --> G
        G --> H[Cross-Encoder Reranker\nms-marco-MiniLM-L-6-v2]
        H --> I[Top-3 Relevant Context Snippets]
        I --> J{LLM Provider Selection}
        J -->|Local Ollama| K1[phi4-mini / llama3]
        J -->|Cloudflare Workers AI API| K2[@cf/meta/llama-3.1-8b-instruct]
        K1 --> L[Formatted Answer with Code Snippet References]
        K2 --> L
    end
```

---

## 🗺️ Development Phases & Roadmap

| Phase | Module | Description | Status |
| :--- | :--- | :--- | :---: |
| **Phase 1** | **Text RAG Baseline** | Git cloning, AST language-aware chunking, BGE-small embeddings, Qdrant vector storage, Cross-Encoder reranking, local Ollama execution. | ✅ Active / Implemented |
| **Phase 2** | **Cloudflare Workers AI API Integration** | Modular REST API LLM provider using `@cf/meta/llama-3.1-8b-instruct` (optimized to prevent free tier neuron depletion). | ✅ Active / Implemented |
| **Phase 3** | **Multimodal Image & Diagram RAG** | Parsing repo diagrams, screenshots, `.png`, `.jpg`, `.svg` files via CLIP vector embeddings (`clip-ViT-B-32`) and Vision LLM text summaries. | ⏳ Planned |
| **Phase 4** | **Unified CLI & Function Interface** | Unified execution entry point eliminating detached script calls (`ingest.py` + `generate.py`). | ⏳ Planned |
| **Phase 5** | **Multi-Turn Chat Memory** | Session message buffer and history-aware query rewriting for conversational context. | ⏳ Planned |
| **Phase 6** | **RAG Evaluation Metrics & Benchmarks** | Benchmark metrics tools measuring **Hit Rate@K**, **MRR@K**, and **Latency breakdown**. | ⏳ Planned |
| **Phase 7** | **Interactive Web UI & Dashboard** | Dark-mode web interface with live chat, model toggling, and vector database status. | ⏳ Planned |

---

## ⚙️ Features Implemented

- **⚡ AST Language-Aware Code Chunking**: Recursively splits Python, JavaScript, TypeScript, C++, Java, Go, and Markdown files preserving code structures.
- **🎯 Two-Stage Retrieval & Reranking**: Combines **BGE-small-en-v1.5** vector search with **ms-marco-MiniLM-L-6-v2** Cross-Encoder reranking for top precision.
- **🌐 Dual LLM Providers**:
  - **Local Ollama**: Run locally using `phi4-mini` or `llama3`.
  - **Cloudflare Workers AI REST API**: Route requests to `@cf/meta/llama-3.1-8b-instruct` using Cloudflare's free tier.

---

## 🛠️ Setup & Execution Guide

### 1. Prerequisites
- Python 3.10+
- Qdrant Vector Database running locally on `http://localhost:6333` (e.g. `docker run -p 6333:6333 qdrant/qdrant`)

### 2. Environment Configuration (`.env`)
Create a `.env` file in the project root:
```env
# Qdrant Settings
QDRANT_URL="http://localhost:6333"
DEFAULT_COLLECTION_NAME="github_codebase"

# Provider Settings ("cloudflare" or "ollama")
DEFAULT_LLM_PROVIDER="cloudflare"
OLLAMA_MODEL="phi4-mini"

# Cloudflare Workers AI API Credentials
CLOUDFLARE_ACCOUNT_ID="your_cloudflare_account_id"
CLOUDFLARE_API_TOKEN="your_cloudflare_api_token"
CLOUDFLARE_MODEL="@cf/meta/llama-3.1-8b-instruct"
```

### 3. Usage Steps

#### Step 1: Ingest a GitHub Repository
Run `ingest.py` to clone and index a repo into Qdrant:
```bash
python ingest.py
```

#### Step 2: Query the Codebase
Run `generate.py` to ask questions about the indexed codebase:
```bash
python generate.py
```

---

## ☁️ Cloudflare Workers AI API Setup Guide

To use Cloudflare Workers AI (10,000 free neurons daily):
1. Log in to the [Cloudflare Dashboard](https://dash.cloudflare.com/).
2. Navigate to **Workers & Pages** $\rightarrow$ **Workers AI**.
3. Copy your **Account ID**.
4. Go to **API Tokens** $\rightarrow$ **Create Token** $\rightarrow$ select **Workers AI (Read)** permissions.
5. Set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` in your `.env` file or environment variables.

---

## 📝 License
Distributed under the MIT License. Built by [Mahesh Paul J](https://github.com/maheshpaulj).
