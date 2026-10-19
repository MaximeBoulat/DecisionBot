import dataclasses
import json

from Agent.agent import Agent
from Agent.code_executor import CodeExecutor
from Agent.jev_decision_model import SystemOneDecisionModel
from Agent.openai_generator import OpenAICodeGenerationModel
from config import (
    DATA_DIR,
    JEV_MODEL,
    JEV_URL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_URL,
    OUT_PATH,
    SCRATCH_DIR,
    TYPESAFE_API_KEY,
)
from data import DabstepData
from models import EpisodeResult


def _summarize(result: EpisodeResult) -> str:
    llm_steps = [s for s in result.steps if s.decided_by == "llm"]
    jev_steps = [s for s in result.steps if s.decided_by == "jev"]
    errors = sum(1 for s in llm_steps if s.stderr)
    lines = [
        f"Task {result.task_id}",
        f"  outcome: {'finished' if result.finished else 'not finished'}"
        + (f" ({result.error})" if result.error else ""),
        f"  final answer: {result.final_answer!r}",
        f"  steps: {len(result.steps)} total, {len(llm_steps)} code ({errors} errored), "
        f"{len(jev_steps)} stop attempts rejected by jev",
    ]
    if jev_steps:
        proposals = ", ".join(f"{s.proposed_answer!r}@{s.jev_confidence}" for s in jev_steps)
        lines.append(f"  rejected proposals: {proposals}")
    return "\n".join(lines)


def _summarize_run(results: list[EpisodeResult]) -> str:
    finished = sum(1 for r in results if r.finished)
    total_steps = sum(len(r.steps) for r in results)
    code_errors = sum(1 for r in results for s in r.steps if s.decided_by == "llm" and s.stderr)
    jev_rejections = sum(1 for r in results for s in r.steps if s.decided_by == "jev")
    return "\n".join([
        f"Run complete: {len(results)} tasks",
        f"  finished: {finished}, not finished: {len(results) - finished}",
        f"  total steps: {total_steps} ({total_steps / len(results):.1f} avg per task)",
        f"  code errors: {code_errors}",
        f"  jev rejections: {jev_rejections}",
    ])


def main() -> None:
    data = DabstepData(DATA_DIR)
    data.fetch()

    tasks = list(data.load_tasks().values())
    documentation = data.load_documentation()
    data_file_paths = data.data_file_paths()

    generator = OpenAICodeGenerationModel(OPENAI_MODEL, OPENAI_API_KEY, documentation, data_file_paths, OPENAI_URL)
    decision_model = SystemOneDecisionModel(JEV_MODEL, JEV_URL, api_key=TYPESAFE_API_KEY)
    executor = CodeExecutor(SCRATCH_DIR)
    agent = Agent(generator, decision_model, executor, documentation)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results: list[EpisodeResult] = []
    for i, task in enumerate(tasks, 1):
        print(f"executing task {i}/{len(tasks)} (task_id={task.task_id})")
        try:
            result = agent.run(task)
        except Exception as e:
            result = EpisodeResult(task.task_id, (), None, False, f"{type(e).__name__}: {e}")
        print(_summarize(result))
        results.append(result)
        with open(OUT_PATH, "a") as f:
            f.write(json.dumps(dataclasses.asdict(result)) + "\n")

    print()
    print(_summarize_run(results))


if __name__ == "__main__":
    main()
