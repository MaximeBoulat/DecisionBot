"""Step-level labels for verifier evaluation. Evaluation only: nothing here reaches a verifier state.

A final answer is labeled by the Scorer. A compute_fees call on a fee question with a fixed shape is labeled
against the scope the question implies (merchant, period, filters, hypothetical change). Seeded steps copy a
correctly labeled step and introduce one known fault.
"""
from __future__ import annotations

import calendar
import json
import re

from models import AgentStep, EpisodeResult, Task
from Scorer.scorer import Scorer, same_answer
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


def _numbers(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _numbers(item, f"{path}.{key}" if path else key)
    elif isinstance(value, str) and re.fullmatch(r"-?\d+(\.\d+)?", value):
        yield path, value


def seeds(point: Point) -> list[tuple[Point, str]]:
    """Copies of a correctly labeled step, each with one known fault and the same history."""
    out = []
    if point.tool_name == "compute_fees":
        args = point.tool_args
        if isinstance(args.get("month"), int):
            out.append(({**args, "month": args["month"] % 12 + 1}, "period"))
        elif isinstance(args.get("day_of_year_min"), int):
            shift = -1 if args["day_of_year_min"] > 1 else 1
            out.append(({**args, "day_of_year_min": args["day_of_year_min"] + shift,
                         "day_of_year_max": args["day_of_year_max"] + shift}, "period"))
        else:
            out.append(({**args, "month": 1}, "period"))
        merchant = args.get("merchant")
        out.append(({**args, "merchant": MERCHANTS[(MERCHANTS.index(merchant) + 1) % len(MERCHANTS)]
                     if merchant in MERCHANTS else MERCHANTS[0]}, "merchant"))
        if not args.get("exclude_refused"):
            out.append(({**args, "exclude_refused": True}, "filter"))
        scenario = args.get("scenario")
        for key in ("fee_rule_id", "rate", "merchant_category_code"):
            if scenario and isinstance(scenario.get(key), (int, float)) and not isinstance(scenario[key], bool):
                out.append(({**args, "scenario": {**scenario, key: scenario[key] + 1}}, "change"))
                break
        return [(Point(f"{point.point_id}:seed-{kind}", point.task, point.history, point.tool_name, new), kind)
                for new, kind in out]
    if point.tool_name == "finish" and re.search(r"\d", point.final_answer or ""):
        out.append(("Not Applicable", "give_up"))
        fee_results = [s for s in point.history if s.tool_name == "compute_fees"]
        if fee_results and re.fullmatch(r"-?[\d.]+", point.final_answer.strip()):
            try:
                result = json.loads(fee_results[-1].tool_result)
            except ValueError:
                result = {}
            decimals = len(point.final_answer.strip().partition(".")[2])
            summary = {k: result[k] for k in ("baseline", "scenario", "delta_fee") if k in result}
            for path, value in _numbers(summary):
                if path.endswith(("total_fee", "delta_fee", "matched_fee_total")):    # the last: runs before the policy fix
                    other = f"{float(value):.{decimals}f}"
                    if not same_answer(other, point.final_answer):
                        out.append((other, "quantity"))
                        break
        return [(Point(f"{point.point_id}:seed-{kind}", point.task, point.history, "finish", {}, answer), kind)
                for answer, kind in out]
    return []
