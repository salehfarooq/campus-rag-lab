from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheLookup:
    response: str | None
    hit: bool
    similarity: float


class SemanticResponseCache:
    def __init__(self, threshold: float = 0.82) -> None:
        self.threshold = threshold
        self.hits = 0
        self.misses = 0
        self._queries: list[str] = []
        self._responses: list[str] = []
        self._vectors: list[Counter[str]] = []

    def lookup(self, query: str) -> CacheLookup:
        if not self._queries:
            self.misses += 1
            return CacheLookup(response=None, hit=False, similarity=0.0)

        query_vector = _char_ngrams(query)
        scores = [_cosine(query_vector, vector) for vector in self._vectors]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        best_score = scores[best_index]

        if best_score >= self.threshold:
            self.hits += 1
            return CacheLookup(response=self._responses[best_index], hit=True, similarity=best_score)

        self.misses += 1
        return CacheLookup(response=None, hit=False, similarity=best_score)

    def store(self, query: str, response: str) -> None:
        self._queries.append(query.lower().strip())
        self._responses.append(response)
        self._vectors.append(_char_ngrams(query))


def _char_ngrams(text: str) -> Counter[str]:
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    grams: Counter[str] = Counter()
    for ngram_size in range(3, 6):
        for index in range(max(len(normalized) - ngram_size + 1, 0)):
            grams[normalized[index : index + ngram_size]] += 1
    return grams


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0
