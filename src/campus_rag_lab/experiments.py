from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from campus_rag_lab.config import Settings
from campus_rag_lab.generation import CampusRagService, GeminiGroundedGenerator
from campus_rag_lab.ingestion import Document, chunk_documents, load_qa_csv
from campus_rag_lab.retrieval import LocalVectorStore


@dataclass(frozen=True)
class EvaluationRow:
    question: str
    expected: str
    answer: str
    correct: bool
    abstained: bool
    latency_seconds: float
    cache_hit: bool
    top_score: float
    chunk_size: int
    overlap: int
    top_k: int
    threshold: float


def evaluate(
    source_documents: list[Document],
    questions_csv: str | Path,
    settings: Settings,
    chunk_sizes: list[int],
    overlaps: list[int],
    top_k_values: list[int],
    thresholds: list[float],
) -> list[EvaluationRow]:
    questions = load_qa_csv(questions_csv)
    rows: list[EvaluationRow] = []

    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            store = LocalVectorStore()
            store.add_documents(chunk_documents(source_documents, chunk_size=chunk_size, overlap=overlap))
            service = CampusRagService(store, GeminiGroundedGenerator(settings))

            for top_k in top_k_values:
                for threshold in thresholds:
                    for item in questions:
                        trace = service.ask(item["question"], top_k=top_k, threshold=threshold)
                        expected = item["answer"].strip().lower()
                        answer = trace.answer.strip().lower()
                        abstained = answer == "i don't know"
                        correct = answer == expected or (expected == "i don't know" and abstained)
                        top_score = trace.retrieved[0].score if trace.retrieved else 0.0
                        rows.append(
                            EvaluationRow(
                                question=item["question"],
                                expected=item["answer"],
                                answer=trace.answer,
                                correct=correct,
                                abstained=abstained,
                                latency_seconds=trace.latency_seconds,
                                cache_hit=trace.cache_hit,
                                top_score=top_score,
                                chunk_size=chunk_size,
                                overlap=overlap,
                                top_k=top_k,
                                threshold=threshold,
                            )
                        )
    return rows


def write_rows(rows: list[EvaluationRow], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

