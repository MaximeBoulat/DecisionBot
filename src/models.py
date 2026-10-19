from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Task:
    task_id: int
    question: str
    answer_format: str


@dataclass(frozen=True)
class AgentStep:
    index: int
    tool_name: str
    tool_args: dict
    tool_result: str
    decided_by: Literal["llm", "jev"]
    proposed_answer: str | None = None
    jev_confidence: float | None = None
    accepted: bool | None = None
    error: str | None = None
    decision_question: DecisionQuestion | None = None
    decision_threshold: float | None = None
    decisions: tuple[DecisionRecord, ...] = ()
    argument_sources: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeResult:
    task_id: int
    steps: tuple[AgentStep, ...]
    final_answer: str | None
    finished: bool
    error: str | None
    task: Task | None = None
    documentation: dict[str, str] = field(default_factory=dict)
    mode: str = "cascade"
    llm_calls: int = 0
    jev_calls: int = 0


@dataclass(frozen=True)
class DecisionQuestion:
    name: str
    kind: Literal["noul", "choice"]
    instructions: str
    criteria: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.kind == "choice" and not self.criteria:
            raise ValueError(f"question {self.name!r}: a choice question needs criteria")
        if self.kind == "noul" and self.criteria:
            raise ValueError(f"question {self.name!r}: a noul question takes no criteria")


@dataclass(frozen=True)
class DecisionRequest:
    questions: tuple[DecisionQuestion, ...]

    def names(self) -> tuple[str, ...]:
        return tuple(q.name for q in self.questions)


@dataclass(frozen=True)
class QuestionAnswer:
    probability: float | None = None
    probabilities: dict[str, float] | None = None
    choice: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class DecisionResponse:
    answers: dict[str, QuestionAnswer]
    input_tokens: int | None = None
    model: str | None = None


@dataclass(frozen=True)
class DecisionRecord:
    point: str
    question: DecisionQuestion
    choice: str | None
    confidence: float | None
    threshold: float | None
    accepted: bool
    response: DecisionResponse | None = None
    error: str | None = None
