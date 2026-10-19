from __future__ import annotations

import json
from dataclasses import dataclass

from models import AgentStep, Task


def truncate(text: str, limit: int) -> str:
    marker = "...[truncated]..."
    if len(text) > limit:
        available = limit - len(marker)
        text = text[:available // 2] + marker + text[-(available - available // 2):]
    return text


@dataclass(frozen=True)
class DecisionContext:
    task: Task
    documentation: dict[str, str]
    history: tuple[AgentStep, ...]

    @property
    def state(self) -> str:
        trace = "\n".join(
            f"[{step.index}] {step.tool_name}({json.dumps(step.tool_args)}) -> "
            f"{truncate(step.tool_result, 400)}"
            for step in self.history[-8:]
        )
        observation = truncate(self.history[-1].tool_result, 2000) if self.history else "No observations yet."
        docs = "\n\n".join(f"{name}:\n{text}" for name, text in self.documentation.items())
        return truncate(
            f"TASK: {self.task.question}\nANSWER FORMAT: {self.task.answer_format}\n"
            f"DOCUMENTATION:\n{docs}\nHISTORY:\n{trace}\nCURRENT OBSERVATION:\n{observation}",
            12000,
        )
