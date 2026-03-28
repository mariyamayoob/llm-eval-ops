from __future__ import annotations

import re
from statistics import mean

from model.contracts import RetrievedChunk, RetrievalStats
from services.kb_service import KBService


TOKEN_RE = re.compile(r"[a-z0-9]+")


class RetrievalService:
    def __init__(self, kb_service: KBService, top_k: int = 3) -> None:
        self.kb_service = kb_service
        self.top_k = top_k

    def retrieve(self, question: str, scenario: str) -> tuple[list[RetrievedChunk], RetrievalStats]:
        if scenario == "retrieval_miss":
            return [], RetrievalStats(
                top_k=self.top_k,
                candidate_count=0,
                similarity_min=0.0,
                similarity_max=0.0,
                similarity_mean=0.0,
                retrieval_empty=True,
            )

        question_tokens = self._tokenize(question)
        scored: list[tuple[float, dict[str, object]]] = []
        for chunk in self.kb_service.scenario_chunks(scenario):
            searchable = f"{chunk['title']} {chunk['body']} {' '.join(chunk['tags'])}"
            chunk_tokens = self._tokenize(searchable)
            overlap = len(question_tokens & chunk_tokens)
            union = len(question_tokens | chunk_tokens) or 1
            score = overlap / union
            if any(tag in question_tokens for tag in chunk["tags"]):
                score += 0.12
            if score > 0:
                scored.append((round(score, 4), chunk))

        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        top = scored[: self.top_k]
        results = [
            RetrievedChunk(
                id=chunk["id"],
                title=chunk["title"],
                score=score,
                tags=list(chunk["tags"]),
            )
            for score, chunk in top
        ]
        scores = [score for score, _ in scored]
        stats = RetrievalStats(
            top_k=self.top_k,
            candidate_count=len(scored),
            similarity_min=min(scores) if scores else 0.0,
            similarity_max=max(scores) if scores else 0.0,
            similarity_mean=round(mean(scores), 4) if scores else 0.0,
            retrieval_empty=not results,
        )
        return results, stats

    def _tokenize(self, text: str) -> set[str]:
        return set(TOKEN_RE.findall(text.lower()))
