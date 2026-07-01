"""
retrieval/vector_store.py
----------------------------
Persists chunks + their embeddings in the `chunks` SQLite table, and
performs cosine-similarity search over a user's chunks. For app-scale
data volumes, loading a user's vectors into memory for search is fast
enough — no separate vector DB needed. The interface below is narrow
enough that swapping in FAISS/Chroma later only touches this file.
"""

import sqlite3
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_utils import get_connection


def _serialize(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _deserialize(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def add_chunks(document_id: int, user_id: int, chunks: list[str], embeddings: np.ndarray) -> list[int]:
    """Store chunk texts + embeddings for a document. Returns list of new chunk ids."""
    conn = get_connection()
    ids = []
    try:
        for idx, (content, emb) in enumerate(zip(chunks, embeddings)):
            cur = conn.execute(
                """INSERT INTO chunks (document_id, user_id, chunk_index, content, embedding)
                   VALUES (?, ?, ?, ?, ?)""",
                (document_id, user_id, idx, content, _serialize(emb)),
            )
            ids.append(cur.lastrowid)
        conn.commit()
        return ids
    finally:
        conn.close()


def get_chunks_for_document(document_id: int) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, chunk_index, content FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_chunks_for_user(user_id: int) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT c.id, c.document_id, c.chunk_index, c.content, d.filename
               FROM chunks c JOIN documents d ON c.document_id = d.id
               WHERE c.user_id = ? ORDER BY c.document_id, c.chunk_index""",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def search(query_embedding: np.ndarray, user_id: int, top_k: int = 4) -> list[dict]:
    """
    Cosine-similarity search over all chunks belonging to `user_id`.
    Returns top_k chunk dicts sorted by descending similarity score.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT c.id, c.document_id, c.chunk_index, c.content, c.embedding, d.filename
               FROM chunks c JOIN documents d ON c.document_id = d.id
               WHERE c.user_id = ?""",
            (user_id,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    q = query_embedding.astype(np.float32)
    q_norm = np.linalg.norm(q) + 1e-8

    scored = []
    for row in rows:
        vec = _deserialize(row["embedding"])
        v_norm = np.linalg.norm(vec) + 1e-8
        sim = float(np.dot(q, vec) / (q_norm * v_norm))
        scored.append(
            {
                "id": row["id"],
                "document_id": row["document_id"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "filename": row["filename"],
                "score": sim,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
