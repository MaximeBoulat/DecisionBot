from __future__ import annotations

from typing import Protocol

from Agent.context import DecisionContext
from models import DecisionRequest, DecisionResponse


class DecisionModel(Protocol):
    def invoke(self, context: DecisionContext, request: DecisionRequest) -> DecisionResponse: ...
