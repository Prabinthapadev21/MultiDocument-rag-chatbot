"""
services/document_service.py
-------------------------------
The orchestration layer the Streamlit UI calls for "upload a file".
Wires together: loaders -> chunker -> embedder -> vector_store,
and also persists the parsed raw text of the source document.

Flow:
  PDF        -> load_pdf_bytes (PARSE) -> chunk_text -> embed -> store
  JSON / MD  -> load_*_bytes (already text) -> chunk_text -> embed -> store
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_utils import get_connection
from ingestion.loaders import load_document
from ingestion.chunker import chunk_text
from retrieval.retriever import Retriever


def process_and_store_document(
    file_bytes: bytes,
    filename: str,
    doc_type: str,
    user_id: int,
    retriever: Retriever,
) -> dict:
    """
    Full ingestion pipeline for one uploaded file.
    Returns a summary dict: {document_id, num_chunks, preview_text}
    """
    # 1. Extract plain text (PDF is PARSED here; json/markdown pass through)
    raw_text = load_document(file_bytes, doc_type)
    if not raw_text.strip():
        raise ValueError("यो file बाट कुनै text निकाल्न सकिएन (खाली वा unsupported content)।")

    # 2. Save the document record (with full parsed text) first
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO documents (user_id, filename, doc_type, raw_text) VALUES (?, ?, ?, ?)",
            (user_id, filename, doc_type, raw_text),
        )
        document_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()

    # 3. Chunk the text
    chunks = chunk_text(raw_text)
    if not chunks:
        raise ValueError("Text चयन गरियो तर chunk बनाउन सकिएन।")

    # 4. Embed + persist chunks
    embeddings = retriever.embed_texts(chunks)
    from retrieval import vector_store

    chunk_ids = vector_store.add_chunks(document_id, user_id, chunks, embeddings)

    return {
        "document_id": document_id,
        "num_chunks": len(chunks),
        "chunk_ids": chunk_ids,
        "preview_text": raw_text[:500],
    }


def list_documents_for_user(user_id: int) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, filename, doc_type, uploaded_at FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
