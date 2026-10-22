from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from models import EpisodeResult

THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?%?$")
# Interpretation-name parts that depart from the manual's reading. The manual says a null field applies to
# all values, and officially graded DABstep answers agree with empty lists matching all values, every
# matching rule's fee summed, and 'immediate' capture matching only 'immediate'.
OFF_MANUAL = {"mean_matching_rules", "minimum_matching_rule", "maximum_matching_rule",
              "immediate_also_matches_numeric_range"}


def _parts(answer: str) -> list[tuple[str, float | str, int]]:
    text = re.sub(r"\baci\s+", "", re.sub(r"[\[\]'\"`€]", "", answer).strip().lower())
    pieces = [text.replace(",", "")] if THOUSANDS.match(text) else text.split(",")
    parts = []
    for piece in pieces:
        label, _, value = piece.rpartition(":")
        label, value = re.sub(r"\s+", "", label), value.strip()
        if not label and not value:
            continue
        fraction = value.rstrip("%").partition(".")[2]
        try:
            parts.append((label, float(value.rstrip("%")), max(len(fraction.rstrip("0")), 2) if fraction else 0))
        except ValueError:
            parts.append((label, re.sub(r"\s+", "", value), 0))
    return sorted(parts, key=lambda part: (part[0], isinstance(part[1], str), part[1]))


def same_answer(submitted: str, expected: str) -> bool:
    """Equal once brackets, quotes, euro signs, 'ACI ' prefixes, spacing, case and list order are ignored.
    A number matches when the expected value rounds to it at the submitted precision, or within 0.01%.
    Trailing zeros past the second decimal are padding: 7.790000 reads as 7.79, 0.000000 as 0.00."""
    got, want = _parts(submitted), _parts(expected)
    if len(got) != len(want):
        return False
    for (label, value, decimals), (want_label, want_value, _) in zip(got, want):
        if label != want_label:
            return False
        if isinstance(value, float) and isinstance(want_value, float):
            tolerance = max(1e-4 * max(abs(value), abs(want_value)), 0.5 * 10 ** -decimals)
            if abs(value - want_value) > tolerance * (1 + 1e-9):
                return False
        elif value != want_value:
            return False
    return True


def manual_reading(answer: dict[str, str]) -> str | None:
    """The one interpretation consistent with the manual's reading, if exactly one is."""
    names = [name for name in answer
             if name.startswith("empty_lists_wildcard") and not OFF_MANUAL & set(name.split("__"))]
    return names[0] if len(names) == 1 else None


def agreed_answer(answer: dict[str, str]) -> str | None:
    """The answer every interpretation gives, if they all give the same one; then the task is not ambiguous."""
    first, *rest = answer.values()
    return first if all(same_answer(v, first) and same_answer(first, v) for v in rest) else None


@dataclass(frozen=True)
class AnswerKeyEntry:
    answer: str | dict[str, str]
    ambiguous: bool


class Scorer:
    def __init__(self, answer_key_path: Path) -> None:
        self.answer_key = self._load_answer_key(answer_key_path)
        self.official = self._load_official(answer_key_path.with_name("official_readings.json"))

    def _load_official(self, path: Path) -> dict[int, str]:
        """Readings chosen for ambiguous tasks because answers Adyen scored right match only that reading."""
        if not path.exists():
            return {}
        official = {int(task): row["reading"] for task, row in json.loads(path.read_text()).items()}
        for task, reading in official.items():
            entry = self.answer_key.get(task)
            if entry is None or not isinstance(entry.answer, dict) or reading not in entry.answer:
                raise ValueError(f"Official reading {reading!r} is not an interpretation of task {task}")
        return official

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
            answered = result.finished and result.error is None and result.final_answer is not None
            reading = manual_reading(entry.answer) if entry is not None and isinstance(entry.answer, dict) else None
            basis = "manual_reading" if reading else None
            if reading is None and result.task_id in self.official:
                reading, basis = self.official[result.task_id], "official_scores"
            agreed = agreed_answer(entry.answer) if reading is None and entry is not None and isinstance(entry.answer, dict) else None
            if entry is None:
                detail["status"] = "missing_key"
            elif entry.ambiguous and reading is None and agreed is None:
                alternatives = entry.answer if isinstance(entry.answer, dict) else {"unspecified": entry.answer}
                detail["status"] = "ambiguous"
                detail["matched_interpretations"] = [
                    name for name, answer in alternatives.items()
                    if answered and same_answer(result.final_answer, answer)
                ]
            else:
                expected = entry.answer[reading] if reading else agreed if agreed is not None else entry.answer
                matches = answered and same_answer(result.final_answer, expected)
                detail["status"] = "correct" if matches else "incorrect"
                detail["expected_answer"] = expected
                if reading:
                    detail["interpretation"], detail["scoring_basis"] = reading, basis
                elif agreed is not None:
                    detail["scoring_basis"] = "all_interpretations_agree"
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
            "comparison": ("normalized, numbers to submitted precision or 0.01%; fee interpretations by the manual's reading; "
                           "readings with official evidence from official_readings.json; tasks whose readings agree are graded"),
            "tasks": details,
        }
