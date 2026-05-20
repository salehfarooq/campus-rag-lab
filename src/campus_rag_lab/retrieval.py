from __future__ import annotations

import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from campus_rag_lab.ingestion import Document


@dataclass(frozen=True)
class RetrievalResult:
    text: str
    score: float
    metadata: dict[str, str]


class LocalVectorStore:
    """Small, inspectable lexical vector store for portfolio demos and tests."""

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._vectors: list[Counter[str]] = []

    @property
    def documents(self) -> list[Document]:
        return list(self._documents)

    def add_documents(self, documents: list[Document]) -> None:
        self._documents.extend(documents)
        self._vectors = [_vectorize(doc.text) for doc in self._documents]

    def search(self, query: str, top_k: int = 3, threshold: float = 0.1) -> list[RetrievalResult]:
        if not query.strip() or not self._documents:
            return []

        query_vector = _vectorize(query)
        scores = [_cosine(query_vector, doc_vector) for doc_vector in self._vectors]
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

        results: list[RetrievalResult] = []
        for index, score in ranked[:top_k]:
            if float(score) < threshold:
                continue
            doc = self._documents[index]
            results.append(RetrievalResult(text=doc.text, score=float(score), metadata=doc.metadata))
        return results

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: str | Path) -> "LocalVectorStore":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


def _vectorize(text: str) -> Counter[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return Counter(token for token in tokens if len(token) > 2)


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0
