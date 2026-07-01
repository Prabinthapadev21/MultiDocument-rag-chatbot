"""
ingestion/loaders.py
---------------------
Turns an uploaded file into plain text, ready for chunking.
"""

import json
import io
import tempfile
import os

from llama_parse import LlamaParse
from config import LLAMA_CLOUD_API_KEY


parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    result_type="markdown",   # or "text"
)


def load_markdown_bytes(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def load_json_bytes(file_bytes: bytes) -> str:
    data = json.loads(file_bytes.decode("utf-8", errors="ignore"))
    lines: list[str] = []

    def _flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_prefix = f"{prefix}.{k}" if prefix else str(k)
                _flatten(v, new_prefix)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _flatten(item, f"{prefix}[{i}]")

        else:
            lines.append(f"{prefix}: {obj}")

    _flatten(data)
    return "\n".join(lines)


def load_pdf_bytes(file_bytes: bytes) -> str:
    """
    Parse PDF using LlamaParse.

    Unlike pypdf, LlamaParse preserves document structure,
    tables, headings, and can handle difficult PDFs much better.
    """

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        documents = parser.load_data(tmp_path)

        text = "\n\n".join(doc.text for doc in documents)

        return text

    finally:
        os.remove(tmp_path)


def load_document(file_bytes: bytes, doc_type: str) -> str:
    doc_type = doc_type.lower()

    if doc_type == "json":
        return load_json_bytes(file_bytes)

    elif doc_type in ("markdown", "md"):
        return load_markdown_bytes(file_bytes)

    elif doc_type == "pdf":
        return load_pdf_bytes(file_bytes)

    else:
        raise ValueError(f"Unsupported doc_type: {doc_type}")