# Campus RAG Lab

Campus RAG Lab is a portfolio-grade retrieval-augmented generation project built from a class assignment on handbook QA. It turns the original notebook workflow into a clean Python package with a CLI, Streamlit dashboard, local retrieval,  semantic caching, and reproducible evaluation.


The app includes a **Retrieval X-Ray**: every answer shows retrieved chunks, similarity scores, cache status, latency, and whether the system abstained with `I don't know`. This makes the project about observability and reliability, not just chatbot output.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
make demo
```

## Commands

```bash
campus-rag ingest data/sample/handbook.txt --output artifacts/vector_store.pkl
campus-rag ask "What is the attendance requirement?" --store artifacts/vector_store.pkl
campus-rag evaluate data/sample/handbook.txt data/sample/questions.csv
campus-rag dashboard
```

Install the dashboard extra first when using Streamlit: `pip install -e ".[ui]"`.

## Repository Layout

```text
src/campus_rag_lab/   package code
tests/                unit tests for retrieval, cache, config, and QA behavior
data/sample/          small offline demo dataset
docs/                 architecture and project notes
scripts/              demo automation
```

## Architecture

```text
source docs -> chunking -> local vector store -> retrieval -> Gemini/offline answerer
                                             -> Retrieval X-Ray metrics
question -> semantic cache ------------------/
```

