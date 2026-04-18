from __future__ import annotations

import json
from pathlib import Path


class KBService:
    def __init__(self, kb_path: str = "data/kb.json") -> None:
        self.kb_path = Path(kb_path)
        self._chunks: list[dict[str, object]] | None = None

    def load(self) -> list[dict[str, object]]:
        if self._chunks is None:
            with self.kb_path.open("r", encoding="utf-8-sig") as handle:
                self._chunks = json.load(handle)
        return list(self._chunks)

    def get_chunks(self, ids: list[str]) -> list[dict[str, object]]:
        by_id = {chunk["id"]: chunk for chunk in self.load()}
        return [by_id[chunk_id] for chunk_id in ids if chunk_id in by_id]

    def scenario_chunks(self, scenario: str) -> list[dict[str, object]]:
        chunks = self.load()
        if scenario == "conflicting_evidence":
            return chunks + [
                {
                    "id": "policy-refund-45-manual-review",
                    "title": "Manual review refund exception",
                    "body": "Refund requests may extend to 45 days only when a manual review exception is approved and documented.",
                    "tags": ["refund", "exception", "manual-review"],
                }
            ]
        return chunks
