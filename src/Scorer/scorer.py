from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from models import EpisodeResult


@dataclass(frozen=True)
class AnswerKeyEntry:
    answer: str | dict[str, str]
    ambiguous: bool


class Scorer:
    def __init__(self, answer_key_path: Path) -> None:
        self.answer_key = self._load_answer_key(answer_key_path)

    def _load_answer_key(self, path: Path) -> dict[int, AnswerKeyEntry]:
        key: dict[int, AnswerKeyEntry] = {}
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                answer = row["answer"]
                if not isinstance(answer, (str, dict)):
                    raise ValueError("Answer-key values must be strings or interpretation dictionaries")
                if isinstance(answer, dict) and not all(isinstance(v, str) for v in answer.values()):
                    raise ValueError("Each interpretation must contain a formatted answer string")
                task_id = int(row["task_id"])
                if task_id in key:
                    raise ValueError(f"Duplicate answer-key task {task_id}")
                key[task_id] = AnswerKeyEntry(
                    answer, bool(row.get("ambiguous", False)) or isinstance(answer, dict),
                )
        return key

    def score(self, results: list[EpisodeResult]) -> dict:
        details: list[dict] = []
        for result in results:
            entry = self.answer_key.get(result.task_id)
            detail = {"task_id": result.task_id, "submitted_answer": result.final_answer}
            if entry is None:
                detail["status"] = "missing_key"
            elif entry.ambiguous:
                alternatives = entry.answer if isinstance(entry.answer, dict) else {"unspecified": entry.answer}
                detail["status"] = "ambiguous"
                detail["matched_interpretations"] = [
                    name for name, answer in alternatives.items()
                    if result.finished and result.error is None and result.final_answer == answer
                ]
            else:
                matches = result.finished and result.error is None and result.final_answer == entry.answer
                detail["status"] = "correct" if matches else "incorrect"
                detail["expected_answer"] = entry.answer
            details.append(detail)
        correct = sum(d["status"] == "correct" for d in details)
        graded = sum(d["status"] in ("correct", "incorrect") for d in details)
        return {
            "total": len(results),
            "graded": graded,
            "ambiguous": sum(d["status"] == "ambiguous" for d in details),
            "missing_key": sum(d["status"] == "missing_key" for d in details),
            "correct": correct,
            "accuracy": correct / graded if graded else None,
            "comparison": "exact_string",
            "tasks": details,
        }
