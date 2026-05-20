from __future__ import annotations

import re
import time
from dataclasses import dataclass

from campus_rag_lab.cache import SemanticResponseCache
from campus_rag_lab.config import Settings
from campus_rag_lab.retrieval import LocalVectorStore, RetrievalResult


@dataclass(frozen=True)
class AnswerTrace:
    answer: str
    answerable: bool
    latency_seconds: float
    cache_hit: bool
    cache_similarity: float
    retrieved: list[RetrievalResult]


class GeminiGroundedGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def answer(self, question: str, retrieved: list[RetrievalResult]) -> str:
        context = "\n\n".join(result.text for result in retrieved)
        if not context.strip():
            return "I don't know"

        if not self.settings.google_api_key:
            return self._offline_answer(question, context)

        time.sleep(self.settings.rate_limit_sleep)
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.settings.google_api_key)
            model = genai.GenerativeModel(self.settings.gemini_model)
            prompt = (
                "Answer only from the provided university policy context. "
                "If the context does not answer the question, say exactly: I don't know.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
            response = model.generate_content(prompt)
            return (response.text or "I don't know").strip()
        except Exception:
            return self._offline_answer(question, context)

    @staticmethod
    def _offline_answer(question: str, context: str) -> str:
        question_terms = {
            token
            for token in re.findall(r"[a-zA-Z]{4,}", question.lower())
            if token not in {"what", "where", "when", "does", "about", "policy"}
        }
        sentences = re.split(r"(?<=[.!?])\s+", context)
        scored = []
        for sentence in sentences:
            terms = set(re.findall(r"[a-zA-Z]{4,}", sentence.lower()))
            overlap = len(question_terms & terms)
            if overlap:
                scored.append((overlap, sentence))
        if not scored:
            return "I don't know"
        return max(scored, key=lambda item: item[0])[1].strip()


class CampusRagService:
    def __init__(
        self,
        store: LocalVectorStore,
        generator: GeminiGroundedGenerator,
        cache: SemanticResponseCache | None = None,
    ) -> None:
        self.store = store
        self.generator = generator
        self.cache = cache

    def ask(self, question: str, top_k: int = 3, threshold: float = 0.1) -> AnswerTrace:
        start = time.perf_counter()
        cache_hit = False
        cache_similarity = 0.0
        retrieved = self.store.search(question, top_k=top_k, threshold=threshold)

        if self.cache:
            lookup = self.cache.lookup(question)
            cache_similarity = lookup.similarity
            if lookup.hit and lookup.response is not None:
                cache_hit = True
                answer = lookup.response
            else:
                answer = self.generator.answer(question, retrieved)
                self.cache.store(question, answer)
        else:
            answer = self.generator.answer(question, retrieved)

        answerable = answer.strip().lower() != "i don't know"
        return AnswerTrace(
            answer=answer,
            answerable=answerable,
            latency_seconds=time.perf_counter() - start,
            cache_hit=cache_hit,
            cache_similarity=cache_similarity,
            retrieved=retrieved,
        )

