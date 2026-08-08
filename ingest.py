import base64
import os
import re
import shutil
import stat
import requests
from dotenv import load_dotenv
from git import Repo
from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from retrieve import connect_to_qdrant

load_dotenv()

EXTENSION_MAP = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".h": Language.C,
    ".hpp": Language.CPP,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".cs": Language.CSHARP,
    ".md": Language.MARKDOWN,
    ".html": Language.HTML,
    ".rst": Language.RST,
    ".sh": Language.SOL,
    ".sql": None,
    ".json": None,
    ".yaml": None,
    ".yml": None,
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}

SPECIALIZED_VISION_PROMPT = """Analyze this repository technical image, performance benchmark graph, plot, architecture diagram, or screenshot with extreme engineering precision.

Your detailed output must include:
1. GRAPH & PLOT METRICS:
   - Identify Chart Title, X-axis label, Y-axis label, units of measurement, and Legend keys.
   - Extract numerical benchmark metrics, comparative performance trends (e.g. latency vs accuracy, throughput, execution time, memory usage), and key data points.
2. DENSE TEXT & ANNOTATIONS:
   - Transcribe all visible text labels, data tables, callout notes, and code snippets inside the image.
3. ARCHITECTURE & WORKFLOW:
   - Detail every labeled node, component box, directional arrow, and data pipeline sequence.

Be exhaustive and precise so an AI engineering system can understand all performance data points and technical text from this image."""

_clip_model = None
_ocr_reader = None


def get_clip_model():
    """Lazy loads CLIP model for raw visual image embeddings."""
    global _clip_model
    if _clip_model is None:
        try:
            print("Loading CLIP visual model (clip-ViT-B-32)...")
            _clip_model = SentenceTransformer("clip-ViT-B-32")
        except Exception as e:
            print(f"Warning: Could not load CLIP model ({e}). Using text-only vectors.")
            _clip_model = False
    return _clip_model if _clip_model is not False else None


def get_ocr_reader():
    """Lazy loads EasyOCR engine for exact text extraction from plot/image pixels."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            print("Loading EasyOCR engine for image text extraction...")
            _ocr_reader = easyocr.Reader(["en"], gpu=False)
        except Exception as e:
            print(f"OCR warning ({e}). OCR text extraction will be skipped.")
            _ocr_reader = False
    return _ocr_reader if _ocr_reader is not False else None


def extract_ocr_text_from_image(image_path: str) -> str:
    """Extracts exact word-for-word text labels, numbers, and metrics embedded inside an image."""
    if image_path.lower().endswith(".svg"):
        return ""

    reader = get_ocr_reader()
    if not reader:
        return ""

    try:
        results = reader.readtext(image_path, detail=0)
        extracted_text = " ".join(results).strip()
        if extracted_text:
            print(f"🔤 OCR extracted {len(results)} text labels from '{os.path.basename(image_path)}'")
            return extracted_text
    except Exception as e:
        print(f"OCR extraction failed for {image_path}: {e}")

    return ""


def get_collection_name_for_repo(repo_url: str) -> str:
    """Generates a clean, unique Qdrant collection name based on repository URL."""
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", repo_url.strip().rstrip("/")).lower()
    return f"repo_{sanitized[-45:]}"


def remove_readonly(func, path, exc_info):
    """Clear the read-only attribute and retry deletion (Windows fix for .git files)."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clone_repository(repo_url: str, target_dir: str = "./downloaded_repo") -> str:
    """Clones a remote git repository to the target directory."""
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, onerror=remove_readonly)

    print(f"Cloning {repo_url} into {target_dir}...")
    Repo.clone_from(repo_url, target_dir)
    return target_dir


def filter_repository_files(repo_path: str):
    """Walks directory and filters for code, doc, and image files."""
    ignored_dirs = {
        ".git",
        "node_modules",
        "__pycache__",
        "dist",
        "build",
        ".venv",
        "venv",
        "qdrant_db",
        ".idea",
        ".vscode",
    }
    code_files = []
    image_files = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            full_path = os.path.join(root, file)
            if ext in EXTENSION_MAP:
                code_files.append(full_path)
            elif ext in IMAGE_EXTENSIONS:
                image_files.append(full_path)

    return code_files, image_files


def calculate_line_numbers(content: str, chunk_text: str, search_start_idx: int = 0):
    """Calculates 1-indexed start and end line numbers of a chunk within content."""
    start_char_idx = content.find(chunk_text, search_start_idx)
    if start_char_idx == -1:
        start_char_idx = search_start_idx

    start_line = content.count("\n", 0, start_char_idx) + 1
    end_line = start_line + chunk_text.count("\n")
    return start_line, end_line, start_char_idx + len(chunk_text)


def chunk_code_file(
    file_path: str, repo_root: str, chunk_size: int = 800, chunk_overlap: int = 100
):
    """Splits a single code file into language-aware chunks with line numbers."""
    ext = os.path.splitext(file_path)[1].lower()
    language = EXTENSION_MAP.get(ext)
    rel_path = os.path.relpath(file_path, repo_root).replace("\\", "/")

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {rel_path}: {e}")
        return []

    if not content.strip():
        return []

    if language:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    raw_chunks = splitter.create_documents([content])
    processed_chunks = []
    current_search_idx = 0

    for idx, chunk in enumerate(raw_chunks):
        raw_text = chunk.page_content
        start_line, end_line, current_search_idx = calculate_line_numbers(
            content, raw_text, current_search_idx
        )

        chunk.metadata["file_path"] = rel_path
        chunk.metadata["chunk_id"] = f"{rel_path}_chunk_{idx}"
        chunk.metadata["file_type"] = ext
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["start_line"] = start_line
        chunk.metadata["end_line"] = end_line
        chunk.metadata["is_image"] = False
        chunk.metadata["image_path"] = ""

        header = f"# File: {rel_path} (Lines {start_line}-{end_line})\n"
        chunk.page_content = header + raw_text
        processed_chunks.append(chunk)

    return processed_chunks


def describe_image_with_vision(image_path: str, repo_root: str) -> str:
    """Uses OCR + Cloudflare Vision LLM to extract exact text, numbers, and plot metrics from repository images."""
    rel_path = os.path.relpath(image_path, repo_root).replace("\\", "/")
    
    ocr_text = extract_ocr_text_from_image(image_path)
    ocr_section = f"[EXACT OCR TEXT & METRICS EXTRACTED FROM IMAGE]:\n{ocr_text}\n\n" if ocr_text else ""

    acc_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    token = os.getenv("CLOUDFLARE_API_TOKEN")
    vision_description = ""

    if acc_id and token and not image_path.lower().endswith(".svg"):
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
                if len(img_bytes) <= 2 * 1024 * 1024:
                    base64_image = base64.b64encode(img_bytes).decode("utf-8")
                    url = f"https://api.cloudflare.com/client/v4/accounts/{acc_id}/ai/run/@cf/meta/llama-3.2-11b-vision-instruct"
                    headers = {"Authorization": f"Bearer {token}"}
                    payload = {
                        "prompt": SPECIALIZED_VISION_PROMPT,
                        "image": base64_image,
                    }
                    res = requests.post(url, headers=headers, json=payload, timeout=25)
                    if res.status_code == 200 and res.json().get("success"):
                        vision_description = res.json()["result"]["response"]
                        print(f"📊 Vision LLM description generated for '{rel_path}'")
        except Exception as e:
            print(f"Vision API fallback for {rel_path}: {e}")

    if not vision_description:
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                fmt = img.format
                vision_description = f"Format: {fmt}, Dimensions: {w}x{h} pixels."
        except Exception:
            vision_description = "Image asset."

    full_content = (
        f"# Repository Image File: {rel_path}\n"
        f"{ocr_section}"
        f"[VISUAL & BENCHMARK ANALYSIS]:\n{vision_description}"
    )
    return full_content


def process_image_file(image_path: str, repo_root: str) -> Document:
    """Processes a single image file into a Document chunk with deep Graph/Plot/Text captioning."""
    ext = os.path.splitext(image_path)[1].lower()
    rel_path = os.path.relpath(image_path, repo_root).replace("\\", "/")
    caption = describe_image_with_vision(image_path, repo_root)

    abs_image_path = os.path.abspath(image_path).replace("\\", "/")
    metadata = {
        "file_path": rel_path,
        "chunk_id": f"{rel_path}_image",
        "file_type": ext,
        "chunk_index": 0,
        "start_line": 1,
        "end_line": 1,
        "is_image": True,
        "image_path": abs_image_path,
    }
    return Document(page_content=caption, metadata=metadata)


def process_repository(repo_path: str):
    """Processes all matching code and image files in repository into chunks."""
    code_files, image_files = filter_repository_files(repo_path)
    all_documents = []

    print(f"Processing {len(code_files)} code files and {len(image_files)} image files...")

    for file_path in code_files:
        file_chunks = chunk_code_file(file_path, repo_root=repo_path)
        all_documents.extend(file_chunks)

    for img_path in image_files:
        img_doc = process_image_file(img_path, repo_root=repo_path)
        all_documents.append(img_doc)

    print(
        f"Total source files: {len(code_files) + len(image_files)} | Total Document chunks: {len(all_documents)}"
    )
    return all_documents


def get_qdrant_client() -> QdrantClient:
    """Uses shared singleton connection from retrieve module to avoid database lock conflicts."""
    return connect_to_qdrant()


def store_in_qdrant(documents, collection_name="github_codebase"):
    """Embeds and upserts document chunks into Qdrant using Dual Named Multi-Vectors (text_vector + clip_vector)."""
    text_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    print(f"Loading text embedding model: {text_model_name}...")
    text_embedding_model = SentenceTransformer(text_model_name)

    clip_model = get_clip_model()
    client = get_qdrant_client()
    text_vector_size = text_embedding_model.get_sentence_embedding_dimension() or 384
    clip_vector_size = 512

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text_vector": VectorParams(size=text_vector_size, distance=Distance.COSINE),
            "clip_vector": VectorParams(size=clip_vector_size, distance=Distance.COSINE),
        },
    )

    points = []
    print(f"Generating Dual Multi-Vectors (Text + CLIP Visual) for Qdrant collection '{collection_name}'...")

    for idx, doc in enumerate(documents):
        t_vector = text_embedding_model.encode(doc.page_content).tolist()

        is_image = doc.metadata.get("is_image", False)
        img_path = doc.metadata.get("image_path", "")
        c_vector = [0.0] * clip_vector_size

        if is_image and clip_model and img_path and os.path.exists(img_path) and not img_path.lower().endswith(".svg"):
            try:
                with Image.open(img_path) as PIL_img:
                    c_vector = clip_model.encode(PIL_img.convert("RGB")).tolist()
                    print(f"🖼️ Generated 512-dim CLIP visual vector for '{doc.metadata['file_path']}'")
            except Exception as e:
                print(f"CLIP encoding fallback for {img_path}: {e}")

        point = PointStruct(
            id=idx,
            vector={
                "text_vector": t_vector,
                "clip_vector": c_vector,
            },
            payload={
                "code": doc.page_content,
                "file_path": doc.metadata["file_path"],
                "file_type": doc.metadata["file_type"],
                "chunk_id": doc.metadata["chunk_id"],
                "chunk_index": doc.metadata["chunk_index"],
                "start_line": doc.metadata.get("start_line", 1),
                "end_line": doc.metadata.get("end_line", 1),
                "is_image": is_image,
                "image_path": img_path,
            },
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)
    print(f"Successfully indexed {len(points)} Dual Multi-Vector chunks in Qdrant collection '{collection_name}'!")


if __name__ == "__main__":
    TEST_REPO_URL = "https://github.com/maheshpaulj/ResumeItNow"
    coll_name = get_collection_name_for_repo(TEST_REPO_URL)
    repo_dir = clone_repository(TEST_REPO_URL)
    documents = process_repository(repo_dir)
    store_in_qdrant(documents, collection_name=coll_name)