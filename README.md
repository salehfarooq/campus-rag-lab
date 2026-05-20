# Campus RAG Lab

Campus RAG Lab is a portfolio-grade retrieval-augmented generation project built from a class assignment on handbook QA. It turns the original notebook workflow into a clean Python package with a CLI, Streamlit dashboard, local retrieval, Gemini integration, semantic caching, and reproducible evaluation.

## What It Demonstrates

- Production Python packaging with a `src/` layout, typed modules, tests, linting, and CLI entry points.
- Grounded RAG behavior with retrieval traces, abstention handling, and local offline demos.
- Evaluation thinking: parameter sweeps, latency tracking, cache hit rates, and answerability metrics.
- Secret hygiene: real keys live in `.env`; `.env.example` documents required variables.

## Unique Twist

The app includes a **Retrieval X-Ray**: every answer shows retrieved chunks, similarity scores, cache status, latency, and whether the system abstained with `I don't know`. This makes the project about observability and reliability, not just chatbot output.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
make demo
```

No API key is required for the offline demo. Add `GOOGLE_API_KEY` to `.env` to use Gemini.

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

## Publishing Checklist

- Rotate any API keys that previously appeared in notebooks before sharing publicly.
- Keep `.env` local and commit only `.env.example`.
- Run `make test` and `make lint` before pushing.

## CV Summary

Built a production-style RAG observability lab for university policy QA with local vector retrieval, Gemini-based grounded generation, semantic prompt caching, abstention scoring, CLI tooling, Streamlit dashboards, and reproducible parameter sweeps.
