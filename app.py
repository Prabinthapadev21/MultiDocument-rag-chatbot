"""
app.py
-------
Streamlit entry point. Pulls together:
  - auth (login / register, SQLite-backed)
  - document upload (json / markdown / pdf -> parse -> chunk -> embed -> store)
  - chunk viewer (browse stored chunks)
  - question answering (retrieve -> show retrieved chunks -> LLM answer)

Run with:  streamlit run app.py
"""

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.db_utils import init_db
from auth.auth_service import register_user, authenticate_user
from services.document_service import process_and_store_document, list_documents_for_user
from services.history_service import save_qa, get_history_for_user
from retrieval.retriever import Retriever
from retrieval.vector_store import get_chunks_for_document, get_all_chunks_for_user
from qa.qa_engine import answer_question
from config import TOP_K_DEFAULT

st.set_page_config(page_title="RAG QA System", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------
# Startup: make sure DB exists, and cache one Retriever instance
# ---------------------------------------------------------------------
init_db()


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever()


retriever = get_retriever()

# ---------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []


# =======================================================================
# AUTH SCREEN
# =======================================================================
def render_auth_screen():
    st.title("📚 RAG QA System — Login")

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                ok, user, msg = authenticate_user(username, password)
                if ok:
                    st.session_state.user = user
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("New username")
            new_password = st.text_input("New password", type="password")
            submitted = st.form_submit_button("Register")
            if submitted:
                ok, msg = register_user(new_username, new_password)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)


# =======================================================================
# UPLOAD PAGE
# =======================================================================
def render_upload_page(user_id: int):
    st.header("📤 Document Upload")
    st.caption("JSON / Markdown / PDF अपलोड गर्नुहोस्। PDF लाई पहिले parse गरिन्छ, "
               "JSON/Markdown लाई सिधै chunk गरिन्छ।")

    doc_type = st.selectbox("File type", ["pdf", "json", "markdown"])
    accept_ext = {"pdf": ["pdf"], "json": ["json"], "markdown": ["md", "markdown", "txt"]}[doc_type]

    uploaded_file = st.file_uploader(f"Upload a .{'/'.join(accept_ext)} file", type=accept_ext)

    if uploaded_file is not None and st.button("Process & Store", type="primary"):
        with st.spinner("Parsing → Chunking → Embedding → Storing..."):
            try:
                file_bytes = uploaded_file.read()
                result = process_and_store_document(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                    doc_type=doc_type,
                    user_id=user_id,
                    retriever=retriever,
                )
                st.success(f"✅ Stored! Document ID: {result['document_id']} — {result['num_chunks']} chunks created.")
                with st.expander("Preview of extracted text"):
                    st.text(result["preview_text"])
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.subheader("तपाईंका Documents")
    docs = list_documents_for_user(user_id)
    if not docs:
        st.info("अहिलेसम्म कुनै document upload गरिएको छैन।")
    else:
        for d in docs:
            st.write(f"📄 **{d['filename']}** — type: `{d['doc_type']}` — uploaded: {d['uploaded_at']} (doc id: {d['id']})")


# =======================================================================
# CHUNK VIEWER PAGE
# =======================================================================
def render_chunk_viewer_page(user_id: int):
    st.header("🧩 Chunk Viewer")
    st.caption("Store भएका chunk हरू यहाँ हेर्न सकिन्छ।")

    docs = list_documents_for_user(user_id)
    if not docs:
        st.info("अहिलेसम्म कुनै document upload गरिएको छैन।")
        return

    doc_options = {f"{d['filename']} (id {d['id']})": d["id"] for d in docs}
    choice = st.selectbox("Document छान्नुहोस्", ["-- All documents --"] + list(doc_options.keys()))

    if choice == "-- All documents --":
        chunks = get_all_chunks_for_user(user_id)
    else:
        chunks = get_chunks_for_document(doc_options[choice])

    st.write(f"कुल chunk संख्या: **{len(chunks)}**")
    for c in chunks:
        label = f"Chunk #{c['chunk_index']}"
        if "filename" in c:
            label += f" — {c['filename']}"
        with st.expander(label):
            st.write(c["content"])


# =======================================================================
# QUESTION ANSWERING PAGE
# =======================================================================
def render_qa_page(user_id: int):
    st.header("❓ Question Answering")

    api_key = st.session_state.get("anthropic_api_key", "")
    if not api_key:
        st.warning("Sidebar मा Anthropic API key राख्नुहोस् (उत्तर दिनको लागि चाहिन्छ)।")

    top_k = st.slider("कति chunk retrieve गर्ने (top_k)", min_value=1, max_value=10, value=TOP_K_DEFAULT)
    question = st.text_input("तपाईंको प्रश्न लेख्नुहोस्")

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving relevant chunks..."):
            retrieved = retriever.retrieve(question, user_id=user_id, top_k=top_k)
            st.session_state.last_retrieved = retrieved

        if not retrieved:
            st.warning("कुनै सम्बन्धित chunk फेला परेन। पहिले document upload गर्नुहोस्।")
        else:
            with st.spinner("Generating answer..."):
                try:
                    answer = answer_question(question, retrieved, api_key)
                    st.markdown("### उत्तर")
                    st.write(answer)
                    save_qa(user_id, question, answer, [c["id"] for c in retrieved])
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

    # Always show the last retrieved chunks (view retrieved chunks requirement)
    if st.session_state.last_retrieved:
        st.divider()
        st.subheader("🔍 Retrieved Chunks (यो प्रश्नको लागि प्रयोग भएका chunk हरू)")
        for i, c in enumerate(st.session_state.last_retrieved, start=1):
            with st.expander(f"[chunk {i}] {c.get('filename', '')} — score: {c['score']:.3f}"):
                st.write(c["content"])

    st.divider()
    st.subheader("🕘 QA History")
    history = get_history_for_user(user_id, limit=10)
    for h in history:
        with st.expander(f"Q: {h['question']}  ({h['created_at']})"):
            st.write(f"**Answer:** {h['answer']}")
            st.caption(f"Retrieved chunk ids: {h['retrieved_chunk_ids']}")


# =======================================================================
# MAIN APP SHELL
# =======================================================================
def render_main_app():
    user = st.session_state.user
    st.sidebar.title(f"👋 Hi, {user['username']}")

    st.sidebar.text_input(
        "Anthropic API Key",
        type="password",
        key="anthropic_api_key",
        help="Question answering गर्न यो चाहिन्छ। https://console.anthropic.com मा बनाउन सकिन्छ।",
    )

    page = st.sidebar.radio("Navigation", ["📤 Upload", "🧩 View Chunks", "❓ Ask a Question"])

    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.last_retrieved = []
        st.rerun()

    if page == "📤 Upload":
        render_upload_page(user["id"])
    elif page == "🧩 View Chunks":
        render_chunk_viewer_page(user["id"])
    elif page == "❓ Ask a Question":
        render_qa_page(user["id"])


# =======================================================================
if st.session_state.user is None:
    render_auth_screen()
else:
    render_main_app()
