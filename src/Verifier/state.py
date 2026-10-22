from __future__ import annotations

import calendar
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta

from Agent.context import truncate
from Agent.tool_schemas import TOOLS_SCHEMA
from models import AgentStep, Task
from Scorer.scorer import same_answer

STATE_VERSION = 2       # 2: every tool's description is in the state, as the agent had them
HISTORY_STEPS = 6       # most recent steps shown in full; earlier ones as one line each
RESULT_CHARS = 1500     # per tool result, head and tail kept, for every state alike
DESCRIPTIONS = {tool["name"]: tool["description"] for tool in TOOLS_SCHEMA}


@dataclass(frozen=True)
class Point:
    """One step the verifier is asked about: the history the agent had, and what it proposed."""
    point_id: str
    task: Task
    history: tuple[AgentStep, ...]
    tool_name: str                      # "finish" for a final answer
    tool_args: dict
    final_answer: str | None = None


def _day(year: int, day_of_year: int) -> str:
    return (date(year, 1, 1) + timedelta(days=day_of_year - 1)).strftime("%B %-d, %Y")


def call_facts(args: dict) -> list[str]:
    """Plain-language readings of date arguments, computed in code so the model does no calendar arithmetic."""
    facts = []
    year = args.get("year")
    if isinstance(year, int):
        lo, hi = args.get("day_of_year_min"), args.get("day_of_year_max")
        if isinstance(args.get("month"), int) and 1 <= args["month"] <= 12:
            facts.append(f"month {args['month']} is {calendar.month_name[args['month']]} {year}")
        if isinstance(lo, int) and isinstance(hi, int):
            facts.append(f"days of year {lo} to {hi} are {_day(year, lo)} to {_day(year, hi)}" if lo != hi
                         else f"day of year {lo} is {_day(year, lo)}")
        elif isinstance(lo, int) or isinstance(hi, int):
            facts.append("only one day-of-year bound is set")
        if args.get("month") is None and lo is None and hi is None:
            facts.append(f"no month or day bounds: the whole year {year} is selected")
    return facts


def _leaves(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for i, item in enumerate(value[:50]):
            yield from _leaves(item, f"{path}[{i}]")
    elif isinstance(value, (int, float, str)) and not isinstance(value, bool):
        yield path, value


def answer_facts(answer: str, history: tuple[AgentStep, ...]) -> list[str]:
    """Where each number in the final answer appears in the tool results, found in code."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", "")) if answer else []
    if not numbers:
        return [f"the final answer {answer!r} contains no number"]
    facts = []
    for number in numbers[:5]:
        hits = []
        for step in history:
            try:
                result = json.loads(step.tool_result)
            except ValueError:
                continue
            for path, value in _leaves(result):
                try:
                    if same_answer(number, str(float(value))):
                        hits.append(f"step {step.index} {step.tool_name} result `{path}` = {value}")
                except ValueError:
                    continue
        facts.append(f"{number} matches " + "; ".join(hits[:3]) if hits
                     else f"{number} does not appear in any tool result")
    return facts


def build_state(point: Point, documentation: dict[str, str]) -> tuple[dict, dict]:
    """The verifier's view: what the agent itself had, plus facts computed in code. Never labels or keys.
    Documentation comes first so a server can reuse its cached prefix across states."""
    cut = 0
    history = []
    for step in point.history:
        recent = step.index >= len(point.history) - HISTORY_STEPS
        result = step.tool_result if recent else ""
        shown = truncate(result, RESULT_CHARS)
        cut += len(result) - len(shown)
        history.append({"step": step.index, "tool": step.tool_name, "arguments": step.tool_args,
                        **({"result": shown} if recent else {"result": "(earlier result omitted)"})})
    if point.tool_name == "finish":
        proposed = {"kind": "final_answer", "final_answer": point.final_answer or ""}
        facts = answer_facts(point.final_answer or "", point.history)
    else:
        proposed = {"kind": "tool_call", "tool": point.tool_name,
                    "tool_description": DESCRIPTIONS.get(point.tool_name, ""), "arguments": point.tool_args}
        facts = call_facts(point.tool_args)
    state = {
        "documentation": documentation,
        "tools": DESCRIPTIONS,
        "question": point.task.question,
        "answer_format": point.task.answer_format,
        "history": history,
        "proposed_step": proposed,
        "facts": facts,
    }
    return state, {"result_chars_cut": cut, "steps": len(history)}
