# RAG QA System (Streamlit + SQLite)

Modular Retrieval-Augmented-Generation app with login, multi-format
document ingestion (JSON / Markdown / PDF), chunk inspection, and a
question-answering interface that shows exactly which chunks were
retrieved for each answer.

## Features
- **Login / Register** — SQLite-backed auth, salted PBKDF2 password hashing.
- **Document upload** — JSON, Markdown, PDF.
  - PDF is **parsed** first (text extracted page-by-page via `pypdf`).
  - JSON / Markdown are already text, so they go **directly to chunking**
    (JSON is flattened into readable `key: value` lines first).
- **Chunk viewer** — browse every stored chunk, per document or across all documents.
- **Retrieval-augmented QA** — question → embed → cosine-similarity search
  over your chunks → **retrieved chunks are shown in the UI** → Claude
  generates a grounded answer citing chunk numbers.
- **QA history** — past questions/answers + which chunk ids were used.

## Project layout (modular by responsibility)
```
rag_app/
├── app.py                     # Streamlit UI / entry point
├── config.py                  # all tunables in one place
├── database/
│   ├── schema.sql              # table definitions
│   └── db_utils.py             # connection + init
├── auth/
│   └── auth_service.py         # register / login logic
├── ingestion/
│   ├── loaders.py              # json / markdown / pdf -> text
│   └── chunker.py              # text -> overlapping chunks
├── retrieval/
│   ├── embeddings.py           # BaseEmbedder + HashingEmbedder (default)
│   ├── vector_store.py         # persist + cosine-similarity search
│   └── retriever.py            # embed query + search, one call
├── qa/
│   └── qa_engine.py            # build prompt, call Claude
└── services/
    ├── document_service.py     # orchestrates upload -> parse -> chunk -> embed -> store
    └── history_service.py      # QA history persistence
```

Each layer only talks to the layer below it through a small function
interface, so you can swap pieces independently, e.g.:
- Swap `HashingEmbedder` → `SentenceTransformerEmbedder` in `retrieval/embeddings.py` for higher-quality embeddings.
- Swap SQLite vector search in `vector_store.py` → FAISS/Chroma/pgvector without touching `retriever.py` or the UI.
- Swap the LLM call in `qa_engine.py` for a different provider.

## Setup
```bash
cd rag_app
pip install -r requirements.txt
streamlit run app.py
```

On first run, `app_data.db` (SQLite) is created automatically with all
tables (`users`, `documents`, `chunks`, `qa_history`).

## Using it
1. **Register** a username/password, then **login**.
2. In the sidebar, paste your **Anthropic API key** (needed only for
   the "Ask a Question" step — get one at https://console.anthropic.com).
3. Go to **Upload**, pick file type (`pdf` / `json` / `markdown`),
   upload the file, click **Process & Store**.
4. Go to **View Chunks** to inspect exactly how your document was split.
5. Go to **Ask a Question**, type a question, click **Ask**.
   You'll see the **retrieved chunks** (with similarity scores) and
   the generated answer underneath.

## Notes
- Default embedder (`HashingEmbedder`) needs no model download and
  works fully offline/local — good for getting started fast. It uses
  scikit-learn's hashing-trick vectorizer (word + bigram features).
  For better semantic matching on larger corpora, switch to
  `SentenceTransformerEmbedder` (see `retrieval/embeddings.py`).
- All data (users, documents, chunks, embeddings, QA history) lives in
  the single `app_data.db` SQLite file — easy to back up or inspect
  directly with any SQLite browser.
