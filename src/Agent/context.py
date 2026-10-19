from __future__ import annotations

from models import AgentStep, Task


class DecisionContext:
    def __init__(
        self,
        task: Task,
        documentation: dict[str, str],
        history: tuple[AgentStep, ...],
        proposed_answer: str,
    ) -> None:
        self.task = task
        self.documentation = documentation
        self.history = history
        self.proposed_answer = proposed_answer

    @property
    def question(self) -> str:
        return self.task.question

    @property
    def answer_format(self) -> str:
        return self.task.answer_format

    @property
    def agent_trace(self) -> str:
        return "\n".join(
            f"[step {s.index}] tool: {s.tool_name}({s.tool_args})\nresult: {s.tool_result}"
            for s in self.history
        )
