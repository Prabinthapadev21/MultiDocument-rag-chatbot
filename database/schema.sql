-- schema.sql
-- All tables for auth + document storage + RAG chunks + QA history live
-- in one SQLite file so the whole app is a single portable .db file.

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    filename      TEXT NOT NULL,
    doc_type      TEXT NOT NULL,          -- 'json' | 'markdown' | 'pdf'
    raw_text      TEXT,                   -- full extracted/parsed text
    uploaded_at   TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   INTEGER NOT NULL,
    user_id       INTEGER NOT NULL,
    chunk_index   INTEGER NOT NULL,
    content       TEXT NOT NULL,
    embedding     BLOB NOT NULL,          -- numpy float32 array, serialized
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (document_id) REFERENCES documents(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS qa_history (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id            INTEGER NOT NULL,
    question           TEXT NOT NULL,
    answer             TEXT NOT NULL,
    retrieved_chunk_ids TEXT NOT NULL,    -- comma separated chunk ids
    created_at         TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id)
);
