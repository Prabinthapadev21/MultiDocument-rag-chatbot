"""
services/history_service.py
------------------------------
Saves and reads past question/answer sessions, including which chunk
ids were retrieved for each answer (so "view retrieved chunks" can be
replayed later from history too).
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_utils import get_connection


def save_qa(user_id: int, question: str, answer: str, retrieved_chunk_ids: list[int]) -> None:
    conn = get_connection()
    try:
        ids_str = ",".join(str(i) for i in retrieved_chunk_ids)
        conn.execute(
            "INSERT INTO qa_history (user_id, question, answer, retrieved_chunk_ids) VALUES (?, ?, ?, ?)",
            (user_id, question, answer, ids_str),
        )
        conn.commit()
    finally:
        conn.close()


def get_history_for_user(user_id: int, limit: int = 20) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM qa_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
