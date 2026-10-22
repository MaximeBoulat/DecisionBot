"""Step-level labels for verifier evaluation. Evaluation only: nothing here reaches a verifier state.

A final answer is labeled by the Scorer. A compute_fees call on a fee question with a fixed shape is labeled
against the scope the question implies (merchant, period, filters, hypothetical change).
"""
from __future__ import annotations

import calendar
import re

from models import AgentStep, EpisodeResult, Task
from Scorer.scorer import Scorer
from Verifier.state import Point

MONTHS = {name: number for number, name in enumerate(calendar.month_name) if name}
MERCHANTS = ["Belles_cookbook_store", "Crossfit_Hanna", "Golfclub_Baron_Friso", "Martinis_Fine_Steakhouse", "Rafa_AI"]
DATA_YEAR = 2023        # every DABstep payment is from 2023; questions that name only a month mean 2023


def gold_scope(question: str) -> dict | None:
    """The compute_fees scope a fee question implies, or None when its shape is not one of the fixed templates.
    Keys absent from the result are unconstrained; scenario keys apply only to calls that make a change."""
    q, m = question, None
    month = lambda name: MONTHS[name] if name in MONTHS else None
    if m := re.match(r"For the (\d+)\w\w of the year (\d+), what (?:is the total fees|are the Fee IDs)\D+?(\S+?)(?: should pay)?\?", q):
        return {"merchant": m[3], "year": int(m[2]), "month": None, "day_of_year_min": int(m[1]), "day_of_year_max": int(m[1])}
    if m := re.match(r"What are the total fees \(in euros\) that (\S+) paid in (?:(\w+) )?(\d{4})", q):
        return {"merchant": m[1], "year": int(m[3]), "month": month(m[2]), "day_of_year_min": None, "day_of_year_max": None}
    if m := re.match(r"What (?:were|are) the applicable [Ff]ee IDs for (\S+) in (?:(\w+) )?(\d{4})", q):
        return {"merchant": m[1], "year": int(m[3]), "month": month(m[2]), "day_of_year_min": None, "day_of_year_max": None}
    if m := re.match(r"In (?:the year|(\w+)) (\d{4}) what delta would (\S+) pay if the relative fee of the fee with ID=(\d+) changed to (\d+)", q):
        return {"merchant": m[3], "year": int(m[2]), "month": month(m[1]), "day_of_year_min": None, "day_of_year_max": None,
                "scenario": {"fee_rule_id": int(m[4]), "rate": int(m[5])}}
    if m := re.match(r"Imagine the merchant (\S+) had changed its MCC code to (\d+) before (\d+) started", q):
        return {"merchant": m[1], "year": int(m[3]), "month": None, "day_of_year_min": None, "day_of_year_max": None,
                "scenario": {"merchant_category_code": int(m[2])}}
    if m := re.match(r"Looking at the (?:month of (\w+)|year (\d{4})), to which card scheme should the merchant (\S+) steer", q):
        return {"merchant": m[3], "year": int(m[2]) if m[2] else DATA_YEAR, "month": month(m[1]),
                "day_of_year_min": None, "day_of_year_max": None}
    if m := (re.match(r"For (\S+) in (\w+), if we were to move the fraudulent", q)
             or re.match(r"Looking at the year (\d{4}) and at the merchant (\S+), if we were to move the fraudulent", q)):
        merchant, when = (m[1], m[2]) if not m[1].isdigit() else (m[2], m[1])
        return {"merchant": merchant, "year": int(when) if when.isdigit() else DATA_YEAR, "month": month(when),
                "day_of_year_min": None, "day_of_year_max": None, "scenario": {"fraudulent_only": True}}
    if m := re.match(r"In (\d{4}), which merchants were affected by the Fee with ID (\d+)", q):
        return {"merchant": None, "year": int(m[1]), "month": None, "day_of_year_min": None, "day_of_year_max": None}
    if m := re.match(r"During (\d{4}), imagine if the Fee with ID (\d+) was only applied to account type (\w+)", q):
        return {"merchant": None, "year": int(m[1]), "month": None, "day_of_year_min": None, "day_of_year_max": None,
                "scenario": {"fee_rule_id": int(m[2]), "rule_account_type": m[3]}}
    return None


def call_faults(gold: dict, args: dict) -> list[str]:
    """What in a compute_fees call departs from the scope the question implies. A month given as the
    matching day window is the same selection."""
    faults = []
    for key in ("merchant", "year", "month", "day_of_year_min", "day_of_year_max"):
        if key in gold and args.get(key) != gold[key]:
            if key.startswith("day_of_year") or (key == "month" and gold["month"] and args.get("day_of_year_min")):
                continue
            faults.append(key)
    if "day_of_year_min" in gold and gold["day_of_year_min"] is not None:
        if (args.get("day_of_year_min"), args.get("day_of_year_max")) != (gold["day_of_year_min"], gold["day_of_year_max"]):
            faults.append("day_of_year")
    elif args.get("day_of_year_min") is not None or args.get("day_of_year_max") is not None:
        if not (gold.get("month") and _month_window(gold["year"], gold["month"]) ==
                (args.get("day_of_year_min"), args.get("day_of_year_max"))):
            faults.append("day_of_year")
    if args.get("exclude_refused"):
        faults.append("exclude_refused")
    if args.get("card_scheme") is not None:
        faults.append("card_scheme_filter")
    scenario = args.get("scenario")
    if scenario:
        for key, value in gold.get("scenario", {}).items():
            if scenario.get(key) != value:
                faults.append(f"scenario.{key}")
    return faults


def _month_window(year: int, month: int) -> tuple[int, int]:
    from datetime import date
    start = date(year, month, 1).timetuple().tm_yday
    return start, start + calendar.monthrange(year, month)[1] - 1


def episode_points(episode: dict) -> list[Point]:
    """Every step of a recorded main episode as a verification point. Crashed steps are skipped."""
    task = Task(**episode["task"])
    steps = tuple(AgentStep(s["index"], s["tool_name"], s["tool_args"], s["tool_result"], s.get("decided_by", "llm"),
                            proposed_answer=s.get("proposed_answer"), error=s.get("error"))
                  for s in episode["steps"] if s["tool_name"] != "generator")
    return [Point(f"{task.task_id}:{i}", task, steps[:i], step.tool_name, step.tool_args,
                  step.proposed_answer if step.tool_name == "finish" else None)
            for i, step in enumerate(steps)]


def label(point: Point, scorer: Scorer) -> tuple[str | None, list[str]]:
    """'ok', 'wrong' or None (no exact label), with the reasons."""
    if point.tool_name == "finish":
        detail = scorer.score([EpisodeResult(point.task.task_id, (), point.final_answer, True, None)])["tasks"][0]
        status = detail["status"]
        return ({"correct": "ok", "incorrect": "wrong"}.get(status), [] if status != "incorrect" else ["answer"])
    if point.tool_name != "compute_fees" or (gold := gold_scope(point.task.question)) is None:
        return None, []
    faults = call_faults(gold, point.tool_args)
    return ("wrong" if faults else "ok"), faults

