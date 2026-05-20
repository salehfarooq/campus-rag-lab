# Architecture

Campus RAG Lab uses a small local TF-IDF vector store for fast demos and deterministic tests. Gemini is used when `GOOGLE_API_KEY` is configured; otherwise the app falls back to a deterministic extractive answerer so reviewers can run the project offline.

The important design choice is making retrieval behavior visible. The Streamlit dashboard exposes retrieved chunks, scores, cache hits, latency, and abstention behavior for every query.

