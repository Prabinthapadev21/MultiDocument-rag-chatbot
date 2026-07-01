"""
ingestion/chunker.py
----------------------
Uses LlamaIndex's SentenceSplitter to create semantic chunks.

Unlike a simple word-based sliding window, this preserves:
- sentence boundaries
- paragraphs
- headings
- tables (from LlamaParse markdown output)

This generally improves retrieval quality for RAG systems.
"""

import sys
import os

from llama_index.core.node_parser import SentenceSplitter

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS


splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE_WORDS,
    chunk_overlap=CHUNK_OVERLAP_WORDS,
)


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Split text into semantic chunks.

    The default splitter is created from config values, but custom
    values can still be passed if needed.
    """

    if not text.strip():
        return []

    local_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    return local_splitter.split_text(text)