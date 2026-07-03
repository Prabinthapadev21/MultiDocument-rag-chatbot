"""
ingestion/loaders.py
---------------------
Turns an uploaded file into plain text, ready for chunking.
"""

import json
import os
import tempfile

from llama_parse import LlamaParse
from config import LLAMA_CLOUD_API_KEY


def get_parser() -> LlamaParse:
    """
    Create a LlamaParse instance.

    This avoids import-time failures when the API key
    is missing on deployment platforms like Streamlit Cloud.
    """

    if not LLAMA_CLOUD_API_KEY:
        raise ValueError(
            "LLAMA_CLOUD_API_KEY is missing.\n"
            "Please add it to:\n"
            "1. .env (for local development)\n"
            "2. Streamlit Secrets (for deployment)"
        )

    return LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
    )


def load_markdown_bytes(file_bytes: bytes) -> str:
    """Load Markdown or plain text files."""
    return file_bytes.decode("utf-8", errors="ignore")


def load_json_bytes(file_bytes: bytes) -> str:
    """Flatten JSON into a readable text representation."""

    data = json.loads(
        file_bytes.decode("utf-8", errors="ignore")
    )

    lines: list[str] = []

    def _flatten(obj, prefix: str = "") -> None:

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = (
                    f"{prefix}.{key}"
                    if prefix
                    else str(key)
                )
                _flatten(value, new_prefix)

        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                _flatten(item, f"{prefix}[{index}]")

        else:
            lines.append(f"{prefix}: {obj}")

    _flatten(data)

    return "\n".join(lines)


def load_pdf_bytes(file_bytes: bytes) -> str:
    """
    Parse a PDF using LlamaParse.

    LlamaParse preserves:
    - Headings
    - Tables
    - Structure
    - Complex layouts
    """

    parser = get_parser()

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        documents = parser.load_data(tmp_path)

        return "\n\n".join(
            doc.text
            for doc in documents
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_document(
    file_bytes: bytes,
    doc_type: str,
) -> str:
    """
    Load any supported document type.

    Supported:
    - PDF
    - JSON
    - Markdown (.md)
    - Text (.txt)
    """

    doc_type = doc_type.lower().strip()

    if doc_type == "pdf":
        return load_pdf_bytes(file_bytes)

    if doc_type == "json":
        return load_json_bytes(file_bytes)

    if doc_type in ("md", "markdown", "txt", "text"):
        return load_markdown_bytes(file_bytes)

    raise ValueError(
        f"Unsupported document type: {doc_type}"
    )