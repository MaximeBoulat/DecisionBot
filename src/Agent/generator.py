from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from models import AgentStep, Task


@dataclass(frozen=True)
class ProposedStep:
    action: Literal["run_code", "finish"]
    code: str | None = None
    answer: str | None = None


class CodeGenerationModel(Protocol):
    def propose_step(self, task: Task, history: tuple[AgentStep, ...]) -> ProposedStep: ...
