from __future__ import annotations

import json
import re
from pathlib import Path

from models import EpisodeResult


def _normalize(answer: str) -> str:
    a = (answer or "").strip()
    try:
        return f"{float(a.replace(',', '').rstrip('%')):.6g}"
    except ValueError:
        return re.sub(r"\s+", "", a).lower()


class Scorer:
    def __init__(self, answer_key_path: Path) -> None:
        self.answer_key = self._load_answer_key(answer_key_path)

    def _load_answer_key(self, path: Path) -> dict[int, str]:
        key = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key[int(row["task_id"])] = row["answer"]
        return key

    def score(self, results: list[EpisodeResult]) -> dict:
        graded = [r for r in results if r.task_id in self.answer_key]
        correct = sum(
            1 for r in graded
            if r.final_answer is not None and _normalize(r.final_answer) == _normalize(self.answer_key[r.task_id])
        )
        return {
            "total": len(results),
            "graded": len(graded),
            "correct": correct,
            "accuracy": correct / len(graded) if graded else None,
        }
