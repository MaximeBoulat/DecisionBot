from __future__ import annotations

from pathlib import Path

from Agent.code_executor import CodeExecutor
from Agent.context import DecisionContext
from Agent.generator import CodeGenerationModel
from Agent.jev_decision_model import SystemOneDecisionModel
from models import AgentStep, DecisionQuestion, DecisionRequest, EpisodeResult, Task

STOP_QUESTION = DecisionQuestion(
    name="stop",
    kind="noul",
    instructions="The agent proposes to stop and submit `proposed_answer`. Has the task actually been "
                 "completed, and is this answer correct given `question`, `documentation` and `agent_trace`?",
)


class Agent:
    def __init__(
        self,
        generator: CodeGenerationModel,
        decision_model: SystemOneDecisionModel,
        executor: CodeExecutor,
        documentation: dict[str, str],
        stop_threshold: float = 0.5,
        max_steps: int = 15,
    ) -> None:
        self.generator = generator
        self.decision_model = decision_model
        self.executor = executor
        self.documentation = documentation
        self.stop_threshold = stop_threshold
        self.max_steps = max_steps

    def run(self, task: Task) -> EpisodeResult:
        history: list[AgentStep] = []
        for index in range(self.max_steps):
            print(f"  task {task.task_id} step {index + 1}/{self.max_steps}: asking the generator...")
            proposed = self.generator.propose_step(task, tuple(history))
            if proposed.action == "finish":
                assert proposed.answer is not None
                print(f"  task {task.task_id} step {index + 1}/{self.max_steps}: proposed {proposed.answer!r}, "
                      f"asking jev...")
                context = DecisionContext(task, self.documentation, tuple(history), proposed.answer)
                request = DecisionRequest(questions=(STOP_QUESTION,))
                response = self.decision_model.invoke(context, request)
                confidence = response.answers["stop"].probability
                assert confidence is not None
                accepted = confidence >= self.stop_threshold
                print(f"  task {task.task_id} step {index + 1}/{self.max_steps}: jev confidence={confidence:.2f} "
                      f"-> {'accepted' if accepted else 'rejected'}")
                if accepted:
                    return EpisodeResult(task.task_id, tuple(history), proposed.answer, True, None)
                history.append(AgentStep(
                    index, "", "", "stop rejected by jev", decided_by="jev",
                    proposed_answer=proposed.answer, jev_confidence=confidence,
                ))
                continue
            assert proposed.code is not None
            print(f"  task {task.task_id} step {index + 1}/{self.max_steps}: executing code...")
            stdout, stderr = self.executor.run(proposed.code)
            print(f"  task {task.task_id} step {index + 1}/{self.max_steps}: "
                  f"{'error' if stderr else 'ok'} ({len(stdout)} chars stdout)")
            history.append(AgentStep(index, proposed.code, stdout, stderr, decided_by="llm"))
        return EpisodeResult(task.task_id, tuple(history), None, False, "max steps exceeded")
