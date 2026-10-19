from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from models import AgentStep, Task


@dataclass(frozen=True)
class ProposedStep:
    action: Literal["call_tool", "finish"]
    tool_name: str | None = None
    tool_args: dict | None = None
    answer: str | None = None


class ActionModel(Protocol):
    def propose_step(self, task: Task, history: tuple[AgentStep, ...]) -> ProposedStep: ...
