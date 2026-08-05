import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Qdrant Vector DB Settings
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION_NAME = os.getenv("DEFAULT_COLLECTION_NAME", "github_codebase")

# Model Settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "sentence-transformers/clip-ViT-B-32")

# LLM Provider Defaults
DEFAULT_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "cloudflare")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Cloudflare Workers AI Settings
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_MODEL = os.getenv("CLOUDFLARE_MODEL", "@cf/meta/llama-3.1-8b-instruct")
CLOUDFLARE_VISION_MODEL = os.getenv("CLOUDFLARE_VISION_MODEL", "@cf/meta/llama-3.2-11b-vision-instruct")

# Text Splitter Defaults
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
