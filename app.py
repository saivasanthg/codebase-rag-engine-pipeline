import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
USER_NAME = "Sai Vasanth"

st.set_page_config(
    page_title="Codebase RAG Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Aesthetic Minimalist Dark Slate CSS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin-bottom: 0.15rem;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }

    .status-badge-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #34d399;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    .status-badge-offline {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.25);
        color: #f87171;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-online { background-color: #10b981; }
    .dot-offline { background-color: #ef4444; }

    .repo-card {
        padding: 12px 16px;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        color: #e2e8f0;
        font-size: 0.88rem;
        margin-bottom: 1.25rem;
    }

    .citation-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 4px;
    }

    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.88rem;
        transition: all 0.2s ease;
    }

    div[data-testid="stSidebarNav"] {
        padding-top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_api_connection():
    try:
        res = requests.get(f"{API_BASE_URL}/api/health", timeout=3)
        return res.status_code == 200
    except Exception:
        return False


# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_repo" not in st.session_state:
    st.session_state.ingested_repo = None

if "collection_name" not in st.session_state:
    st.session_state.collection_name = "github_codebase"

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h3 style='font-size: 1.2rem; font-weight: 600; margin-bottom: 0;'>Codebase RAG Engine</h3>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.8rem; color: #94a3b8; margin-top: 4px; line-height: 1.3;'>Multimodal retrieval augmented generation for codebases, architecture diagrams, and metrics</div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 12px 0; border-color: #334155;'>", unsafe_allow_html=True)

    api_online = check_api_connection()
    if api_online:
        st.markdown(
            '<div class="status-badge-online"><span class="status-dot dot-online"></span> FastAPI API Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge-offline"><span class="status-dot dot-offline"></span> FastAPI Offline (localhost:8000)</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #94a3b8; margin-top: 12px; margin-bottom: 8px;'>REPOSITORY INGESTION</div>", unsafe_allow_html=True)
    repo_url_input = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/owner/repository",
        value=st.session_state.ingested_repo or "",
        label_visibility="collapsed",
    )

    if st.button("Ingest Repository", use_container_width=True, type="primary"):
        if not repo_url_input.strip():
            st.error("Please provide a GitHub repository URL.")
        elif not api_online:
            st.error("FastAPI server offline. Start server: uvicorn api:app --reload")
        else:
            with st.spinner("Processing repository..."):
                try:
                    repo_url = repo_url_input.strip()
                    res = requests.post(
                        f"{API_BASE_URL}/api/ingest",
                        json={"repo_url": repo_url},
                        timeout=300,
                    )
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.ingested_repo = repo_url
                        st.session_state.collection_name = data["collection_name"]
                        st.session_state.messages = []
                        st.success(f"Indexed {data['chunks_indexed']} document chunks.")
                    else:
                        st.error(f"Ingestion failed ({res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"Request error: {e}")

    st.markdown("<hr style='margin: 16px 0; border-color: #334155;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #94a3b8; margin-bottom: 8px;'>MODEL CONFIGURATION</div>", unsafe_allow_html=True)

    provider = st.radio("Provider", ["ollama", "cloudflare"], index=0, label_visibility="collapsed")
    cf_account_id = None
    cf_api_token = None

    if provider == "ollama":
        model_name = st.text_input("Ollama Model", value="phi4-mini")
    else:
        model_name = st.selectbox(
            "Cloudflare Model",
            [
                "@cf/meta/llama-3.1-8b-instruct",
                "@cf/meta/llama-3-8b-instruct",
                "@cf/qwen/qwen1.5-7b-chat",
                "@cf/mistral/mistral-7b-instruct-v0.2",
                "@cf/tinyllama/tinyllama-1.1b-chat-v1.0",
            ],
            index=0,
        )
        cf_account_id = st.text_input(
            "Account ID",
            value=os.getenv("CLOUDFLARE_ACCOUNT_ID", ""),
            placeholder="Account ID",
        )
        cf_api_token = st.text_input(
            "API Token",
            type="password",
            value=os.getenv("CLOUDFLARE_API_TOKEN", ""),
            placeholder="API Token",
        )
        if cf_account_id:
            os.environ["CLOUDFLARE_ACCOUNT_ID"] = cf_account_id
        if cf_api_token:
            os.environ["CLOUDFLARE_API_TOKEN"] = cf_api_token

    st.markdown("<hr style='margin: 16px 0; border-color: #334155;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #94a3b8; margin-bottom: 8px;'>RETRIEVAL PARAMS</div>", unsafe_allow_html=True)
    top_k_retrieve = st.slider("Vector Retrieval Candidates", 5, 50, 15)
    top_k_rerank = st.slider("Cross-Encoder Rerank Limit", 1, 10, 3)

    st.markdown("<hr style='margin: 16px 0; border-color: #334155;'>", unsafe_allow_html=True)
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main App ---
st.markdown(f'<div class="main-title">Hello, {USER_NAME}</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Codebase Assistant</div>', unsafe_allow_html=True)

if st.session_state.ingested_repo:
    st.markdown(
        f'<div class="repo-card">Active Repository: <strong>{st.session_state.ingested_repo}</strong> &nbsp;·&nbsp; Collection: <code>{st.session_state.collection_name}</code></div>',
        unsafe_allow_html=True,
    )
else:
    st.info("No repository ingested. Enter a GitHub URL in the sidebar and click Ingest Repository to begin.")


def render_snippets_expander(snippets, standalone_query=None):
    with st.expander("Retrieved Context & Citations"):
        if standalone_query:
            st.caption(f"Search Query: `{standalone_query}`")
        for idx, snip in enumerate(snippets, 1):
            if snip.get("is_image"):
                st.markdown(
                    f'<div class="citation-header">Snippet {idx} (Image Asset) &nbsp;·&nbsp; {snip["file_path"]} &nbsp;·&nbsp; Score: {snip["score"]:.4f}</div>',
                    unsafe_allow_html=True,
                )
                if snip.get("image_path") and os.path.exists(snip["image_path"]):
                    st.image(snip["image_path"], caption=snip["file_path"], use_container_width=True)
                st.caption(snip["code"])
            else:
                line_info = f"Lines {snip['start_line']}-{snip['end_line']}" if snip.get("start_line") else ""
                st.markdown(
                    f'<div class="citation-header">Snippet {idx} &nbsp;·&nbsp; {snip["file_path"]} {line_info} &nbsp;·&nbsp; Score: {snip["score"]:.4f}</div>',
                    unsafe_allow_html=True,
                )
                st.code(
                    snip["code"],
                    language=snip["file_path"].split(".")[-1] if "." in snip["file_path"] else "text",
                )


# Render History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("snippets"):
            render_snippets_expander(msg["snippets"], msg.get("standalone_query"))

# User Input
if prompt := st.chat_input("Ask a question about the repository..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    history_for_llm = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching context & generating response..."):
            try:
                payload = {
                    "query": prompt,
                    "chat_history": history_for_llm,
                    "provider": provider,
                    "model_name": model_name,
                    "collection_name": st.session_state.collection_name,
                    "top_k_retrieve": top_k_retrieve,
                    "top_k_rerank": top_k_rerank,
                    "account_id": cf_account_id,
                    "api_token": cf_api_token,
                }
                res = requests.post(f"{API_BASE_URL}/api/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    snippets = data["snippets"]
                    standalone_query = data["standalone_query"]

                    st.markdown(answer)
                    if snippets:
                        render_snippets_expander(snippets, standalone_query)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "snippets": snippets,
                            "standalone_query": standalone_query,
                        }
                    )
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
