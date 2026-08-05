import os
import shutil
from git import Repo
import stat
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import config

EXTENSION_MAP = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".cpp": Language.CPP,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".md": Language.MARKDOWN
}

def remove_readonly(func, path, exc_info):
    """Clear the read-only attribute and retry deletion (Windows fix for .git files)."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clone_repository(repo_url: str, target_dir: str = "./downloaded_repo") -> str:
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, onerror=remove_readonly)  
    
    print(f"Cloning {repo_url} into {target_dir}...")
    Repo.clone_from(repo_url, target_dir)
    return target_dir

def filter_code_files(repo_path: str):
    ignored_dirs = {".git", "node_modules", "__pycache__", "dist", "build", ".venv", "venv"}
    code_files = []
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in EXTENSION_MAP:
                print(file)
                code_files.append(os.path.join(root, file))
                
    return code_files

def chunk_code_file(file_path: str, repo_root: str, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
    ext = os.path.splitext(file_path)[1].lower()
    language = EXTENSION_MAP.get(ext)
    
    rel_path = os.path.relpath(file_path, repo_root)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {rel_path}: {e}")
        return []

    if language:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    raw_chunks = splitter.create_documents([content])

    processed_chunks = []
    for idx, chunk in enumerate(raw_chunks):
        chunk.metadata["file_path"] = rel_path
        chunk.metadata["chunk_id"] = f"{rel_path}_chunk_{idx}"
        chunk.metadata["file_type"] = ext
        chunk.metadata["chunk_index"] = idx
        
        chunk.page_content = f"# File: {rel_path}\n" + chunk.page_content
        processed_chunks.append(chunk)

    return processed_chunks

def process_repository(repo_path: str):
    code_files = filter_code_files(repo_path)
    all_documents = []
    
    for file_path in code_files:
        file_chunks = chunk_code_file(file_path, repo_root=repo_path)
        all_documents.extend(file_chunks)
        
    print(f"Total source files: {len(code_files)} | Total Document chunks: {len(all_documents)}")
    return all_documents

def store_in_qdrant(documents, collection_name=config.DEFAULT_COLLECTION_NAME, qdrant_url=config.QDRANT_URL):
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    client = QdrantClient(url=qdrant_url)
    vector_size = 384 

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    points = []
    print("Generating embeddings and storing in Qdrant...")

    for idx, doc in enumerate(documents):
        vector = embedding_model.encode(doc.page_content).tolist()
        point = PointStruct(
            id=idx,
            vector=vector,
            payload={
                "code": doc.page_content,
                "file_path": doc.metadata["file_path"],
                "file_type": doc.metadata["file_type"],
                "chunk_id": doc.metadata["chunk_id"],
                "chunk_index": doc.metadata["chunk_index"]
            }
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)
    print(f"Successfully indexed {len(points)} document chunks in Qdrant!")

def ingest_repository(repo_url: str, collection_name: str = config.DEFAULT_COLLECTION_NAME):
    repo_dir = clone_repository(repo_url)
    documents = process_repository(repo_dir)
    store_in_qdrant(documents, collection_name=collection_name)

if __name__ == "__main__":
    TEST_REPO_URL = "https://github.com/maheshpaulj/ResumeItNow"
    ingest_repository(TEST_REPO_URL)