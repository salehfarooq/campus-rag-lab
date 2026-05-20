#!/usr/bin/env bash
set -euo pipefail

campus-rag ingest data/sample/handbook.txt --output artifacts/vector_store.pkl
campus-rag ask "What is the attendance requirement?" --store artifacts/vector_store.pkl

