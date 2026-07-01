"""
retrieval/retriever.py
-------------------------
Thin orchestration layer: embed the query, ask the vector store for the
closest chunks. This is the object the QA engine and the "view
retrieved chunks" UI both call.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOP_K_DEFAULT
from retrieval.embeddings import get_default_embedder
from retrieval import vector_store


class Retriever:
    def __init__(self, embedder=None):
        self._embedder = embedder or get_default_embedder()

    def retrieve(self, query: str, user_id: int, top_k: int = TOP_K_DEFAULT) -> list[dict]:
        """Return top_k relevant chunks (with similarity scores) for a query."""
        query_vec = self._embedder.embed([query])[0]
        return vector_store.search(query_vec, user_id=user_id, top_k=top_k)

    def embed_texts(self, texts: list[str]):
        """Exposed so document_service can embed chunks with the same embedder."""
        return self._embedder.embed(texts)
