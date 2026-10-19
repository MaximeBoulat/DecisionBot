from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from datetime import datetime, timezone

from Agent.agent import Agent
from Agent.cascade import Cascade
from Agent.jev_decision_model import SystemOneDecisionModel
from Agent.openai_generator import OpenAIActionModel
from Agent.tools import ToolsLayer
from Scorer.scorer import Scorer
from config import (
    ANSWER_KEY_PATH, DATA_DIR, JEV_MODEL, JEV_URL, OPENAI_API_KEY,
    OPENAI_MODEL, OPENAI_URL, OUT_PATH, SCORE_PATH, TYPESAFE_API_KEY,
)
from data import DabstepData
from debug_log import DebugLog
from models import EpisodeResult


def _summarize(result: EpisodeResult) -> str:
    return (f"Task {result.task_id}: {'finished' if result.finished else result.error}; "
            f"{len(result.steps)} steps; {result.llm_calls} LLM calls; {result.jev_calls} Jev calls")


def _summarize_run(results: list[EpisodeResult]) -> str:
    return (f"Run complete: {len(results)} tasks; {sum(r.finished for r in results)} finished; "
            f"{sum(r.llm_calls for r in results)} LLM calls; {sum(r.jev_calls for r in results)} Jev calls")


def _score(results: list[EpisodeResult], scorer: Scorer, path: Path = SCORE_PATH) -> None:
    report = scorer.score(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "tasks"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", type=Path, help="Grade a recorded episodes JSONL without model calls.")
    parser.add_argument("--mode", choices=["baseline", "shadow", "cascade"], default="cascade")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--tool-threshold", type=float)
    parser.add_argument("--argument-threshold", type=float)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--task-value-usd", type=float,
                        help="Dollar value of one correct task (V in the cascade's cost model). "
                             "Required in shadow mode; it labels shadow decisions, it is not measured.")
    options = parser.parse_args()
    scorer = Scorer(ANSWER_KEY_PATH)
    if options.score_only is not None:
        results = []
        for line in options.score_only.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                results.append(EpisodeResult(
                    int(row["task_id"]), (), row["final_answer"], row["finished"], row["error"],
                ))
        _score(results, scorer)
    else:
        if not OPENAI_API_KEY or (options.mode != "baseline" and not TYPESAFE_API_KEY):
            raise ValueError("OPENAI_API_KEY is required; shadow/cascade also require TYPESAFE_API_KEY")
        profile = json.loads(options.calibration.read_text()) if options.calibration else None
        tool_threshold, argument_threshold = options.tool_threshold, options.argument_threshold
        if profile is not None:
            if tool_threshold is not None or argument_threshold is not None:
                parser.error("Choose a calibration profile or explicit thresholds, not both")
            if profile["generator_model"] != OPENAI_MODEL or profile["jev_model"] != JEV_MODEL:
                parser.error("Calibration models do not match configured models")
            if profile["confidence_statistic"] != "max_option_probability":
                parser.error("Calibration confidence statistic does not match the cascade")
            tool_threshold, argument_threshold = profile["tool_threshold"], profile["argument_threshold"]
        elif options.mode == "cascade" and (tool_threshold is None or argument_threshold is None):
            parser.error("Cascade requires --calibration or both explicit thresholds; no threshold is assumed")
        if options.mode == "shadow" and options.task_value_usd is None:
            parser.error("Shadow mode requires --task-value-usd to label decisions")
        data = DabstepData(DATA_DIR)
        data.fetch()
        tasks = list(data.load_tasks().values())
        if options.task_ids is not None:
            requested = set(options.task_ids)
            if requested - {task.task_id for task in tasks}:
                parser.error("Unknown task IDs")
            tasks = [task for task in tasks if task.task_id in requested]
        if profile and options.mode == "cascade":
            calibration_ids = set(profile["calibration_task_ids"])
            if options.task_ids and calibration_ids.intersection(options.task_ids):
                parser.error("Evaluation tasks must be separate from calibration tasks")
            tasks = [task for task in tasks if task.task_id not in calibration_ids]
        documentation = data.load_documentation()
        generator = OpenAIActionModel(OPENAI_MODEL, OPENAI_API_KEY, documentation, OPENAI_URL)
        decision_model = SystemOneDecisionModel(JEV_MODEL, JEV_URL, api_key=TYPESAFE_API_KEY)
        tools = ToolsLayer(DATA_DIR)
        cascade = Cascade(decision_model, tool_threshold, argument_threshold)
        agent = Agent(generator, cascade, tools, documentation, options.mode, options.max_steps,
                      scorer, options.task_value_usd, len(tasks))
        run_dir = OUT_PATH.parent / "runs" / (options.mode + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "run.json").write_text(json.dumps({
            "mode": options.mode, "generator_model": OPENAI_MODEL, "jev_model": JEV_MODEL,
            "tool_threshold": tool_threshold, "argument_threshold": argument_threshold,
            "calibrated": profile is not None, "max_steps": options.max_steps,
            "task_ids": [task.task_id for task in tasks],
        }, indent=2) + "\n")
        print(f"Recording run in {run_dir}")

        results: list[EpisodeResult] = []
        for i, task in enumerate(tasks, 1):
            print(f"executing task {i}/{len(tasks)} (task_id={task.task_id})")
            debug_log = DebugLog(run_dir / "debug" / f"task_{task.task_id}.jsonl")
            result = agent.run(task, debug_log, i)
            print(_summarize(result))
            results.append(result)
            with (run_dir / "episodes.jsonl").open("a") as handle:
                handle.write(json.dumps(dataclasses.asdict(result)) + "\n")
        print(_summarize_run(results))
        _score(results, scorer, run_dir / "scores.json")


if __name__ == "__main__":
    main()
