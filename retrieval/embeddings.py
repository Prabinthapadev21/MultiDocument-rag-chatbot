"""
retrieval/embeddings.py
-------------------------
Turns text into fixed-size numeric vectors.

Default implementation: scikit-learn's HashingVectorizer. It needs no
training/fitting step and always produces the same fixed-length vector
for a piece of text, which is exactly what we need for a growing
document store (new chunks can be embedded independently, no re-fit
needed). This keeps the app light — no torch / heavy model download.

If you want higher-quality semantic embeddings later, swap in
SentenceTransformerEmbedder (needs `pip install sentence-transformers`)
or a hosted embedding API — both implement the same BaseEmbedder
interface, so nothing else in the app has to change.
"""

from abc import ABC, abstractmethod
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_DIM


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) float32 numpy array for N input texts."""
        raise NotImplementedError


class HashingEmbedder(BaseEmbedder):
    """Default, dependency-light embedder using the hashing trick + TF-IDF-style weighting."""

    def __init__(self, n_features: int = EMBEDDING_DIM):
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._vectorizer.n_features), dtype=np.float32)
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32)


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Optional higher-quality embedder. Requires:
        pip install sentence-transformers
    Drop-in replacement for HashingEmbedder — same .embed() interface.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self._model.encode(texts, convert_to_numpy=True).astype(np.float32)


def get_default_embedder() -> BaseEmbedder:
    """Factory — change this one line to switch the whole app's embedder."""
    return HashingEmbedder()
