.PHONY: install test lint demo dashboard

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

demo:
	campus-rag ingest data/sample/handbook.txt --output artifacts/vector_store.pkl
	campus-rag ask "What is the attendance requirement?" --store artifacts/vector_store.pkl

dashboard:
	campus-rag dashboard

