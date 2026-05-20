from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from campus_rag_lab.cache import SemanticResponseCache
from campus_rag_lab.config import load_settings
from campus_rag_lab.generation import CampusRagService, GeminiGroundedGenerator
from campus_rag_lab.ingestion import chunk_documents, load_text
from campus_rag_lab.retrieval import LocalVectorStore


st.set_page_config(page_title="Campus RAG Lab", layout="wide")
st.title("Campus RAG Lab")

settings = load_settings()
sample_path = Path("data/sample/handbook.txt")
question_path = Path("data/sample/questions.csv")

with st.sidebar:
    st.header("Retrieval Controls")
    chunk_size = st.slider("Chunk size", 150, 900, 500, 50)
    overlap = st.slider("Overlap", 0, 200, 80, 10)
    top_k = st.slider("Top K", 1, 6, 3)
    threshold = st.slider("Similarity threshold", 0.0, 0.8, 0.1, 0.05)
    use_cache = st.toggle("Semantic cache", value=True)

store = LocalVectorStore()
store.add_documents(chunk_documents(load_text(sample_path), chunk_size=chunk_size, overlap=overlap))
cache = SemanticResponseCache(settings.cache_threshold) if use_cache else None
service = CampusRagService(store, GeminiGroundedGenerator(settings), cache)

tabs = st.tabs(["Ask", "Retrieval X-Ray", "Parameter Sweeps", "Cache Benchmark"])

with tabs[0]:
    question = st.text_input("Question", "What is the attendance requirement?")
    if st.button("Ask", type="primary"):
        trace = service.ask(question, top_k=top_k, threshold=threshold)
        st.metric("Latency", f"{trace.latency_seconds:.3f}s")
        st.metric("Cache hit", str(trace.cache_hit))
        st.metric("Answerable", str(trace.answerable))
        st.subheader("Answer")
        st.write(trace.answer)
        st.session_state["last_trace"] = trace

with tabs[1]:
    trace = st.session_state.get("last_trace")
    if trace:
        for result in trace.retrieved:
            st.progress(min(result.score, 1.0), text=f"{result.score:.3f} similarity")
            st.caption(result.metadata)
            st.write(result.text)
    else:
        st.info("Ask a question first to inspect retrieved chunks.")

with tabs[2]:
    df = pd.read_csv(question_path)
    st.dataframe(df, use_container_width=True)
    st.caption("Run `campus-rag evaluate data/sample/handbook.txt data/sample/questions.csv` for a CSV scorecard.")

with tabs[3]:
    st.write(
        {
            "cache_enabled": use_cache,
            "cache_threshold": settings.cache_threshold,
            "twist": "Cache hits, abstentions, latency, and retrieval scores are visible together.",
        }
    )

