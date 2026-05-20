from __future__ import annotations

import argparse
from pathlib import Path

from campus_rag_lab.cache import SemanticResponseCache
from campus_rag_lab.config import load_settings
from campus_rag_lab.experiments import evaluate, write_rows
from campus_rag_lab.generation import CampusRagService, GeminiGroundedGenerator
from campus_rag_lab.ingestion import chunk_documents, load_pdf, load_text
from campus_rag_lab.retrieval import LocalVectorStore


def _load_documents(path: Path):
    return load_pdf(path) if path.suffix.lower() == ".pdf" else load_text(path)


def _ingest(args: argparse.Namespace) -> None:
    store = LocalVectorStore()
    store.add_documents(
        chunk_documents(_load_documents(args.source), chunk_size=args.chunk_size, overlap=args.overlap)
    )
    store.save(args.output)
    print(f"Indexed {len(store.documents)} chunks -> {args.output}")


def _ask(args: argparse.Namespace) -> None:
    settings = load_settings()
    store = LocalVectorStore.load(args.store)
    cache = SemanticResponseCache(settings.cache_threshold) if args.use_cache else None
    service = CampusRagService(store, GeminiGroundedGenerator(settings), cache)
    trace = service.ask(args.question, top_k=args.top_k, threshold=args.threshold)

    print(f"Answer: {trace.answer}")
    print(
        f"answerable={trace.answerable} latency={trace.latency_seconds:.3f}s "
        f"cache_hit={trace.cache_hit}"
    )
    for result in trace.retrieved:
        source = result.metadata.get("source", "unknown")
        print(f"- score={result.score:.3f} source={source} chunk={result.text[:180]}")


def _evaluate(args: argparse.Namespace) -> None:
    settings = load_settings()
    rows = evaluate(
        _load_documents(args.source),
        args.questions,
        settings,
        chunk_sizes=[300, 500],
        overlaps=[40, 80],
        top_k_values=[2, 3],
        thresholds=[0.05, 0.15],
    )
    write_rows(rows, args.output)
    accuracy = sum(row.correct for row in rows) / len(rows)
    abstention_rate = sum(row.abstained for row in rows) / len(rows)
    print(f"Wrote {len(rows)} rows -> {args.output}")
    print(f"accuracy={accuracy:.2%} abstention_rate={abstention_rate:.2%}")


def _dashboard(_: argparse.Namespace) -> None:
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/campus_rag_lab/dashboard.py"], check=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Campus RAG Lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Index a text or PDF source.")
    ingest.add_argument("source", type=Path)
    ingest.add_argument("--output", "-o", type=Path, default=Path("artifacts/vector_store.pkl"))
    ingest.add_argument("--chunk-size", type=int, default=500)
    ingest.add_argument("--overlap", type=int, default=80)
    ingest.set_defaults(func=_ingest)

    ask = subparsers.add_parser("ask", help="Ask a grounded question.")
    ask.add_argument("question")
    ask.add_argument("--store", type=Path, default=Path("artifacts/vector_store.pkl"))
    ask.add_argument("--top-k", type=int, default=3)
    ask.add_argument("--threshold", type=float, default=0.1)
    ask.add_argument("--no-cache", dest="use_cache", action="store_false")
    ask.set_defaults(use_cache=True, func=_ask)

    evaluate_cmd = subparsers.add_parser("evaluate", help="Run retrieval parameter sweeps.")
    evaluate_cmd.add_argument("source", type=Path)
    evaluate_cmd.add_argument("questions", type=Path)
    evaluate_cmd.add_argument("--output", "-o", type=Path, default=Path("artifacts/evaluation.csv"))
    evaluate_cmd.set_defaults(func=_evaluate)

    dashboard = subparsers.add_parser("dashboard", help="Launch Streamlit dashboard.")
    dashboard.set_defaults(func=_dashboard)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
