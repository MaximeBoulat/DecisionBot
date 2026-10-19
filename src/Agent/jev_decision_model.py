from __future__ import annotations

import json
import urllib.request

from Agent.context import DecisionContext
from Agent.decision_model import DecisionModel
from models import DecisionQuestion, DecisionRequest, DecisionResponse, QuestionAnswer


class SystemOneDecisionModel(DecisionModel):
    def __init__(self, model: str, base_url: str, api_key: str | None = None, timeout: float = 600) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    @property
    def name(self) -> str:
        return self.model

    def _wire_question(self, question: DecisionQuestion) -> dict:
        wire = {"type": question.kind, "instructions": question.instructions}
        if question.criteria:
            wire["criteria"] = question.criteria
        return wire

    def invoke(self, context: DecisionContext, request: DecisionRequest) -> DecisionResponse:
        payload = {
            "model": self.model,
            "state": context.state,
            "questions": {q.name: self._wire_question(q) for q in request.questions},
        }
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.loads(r.read())
        answers: dict[str, QuestionAnswer] = {}
        for name, ans in body["answers"].items():
            if "noul" in ans:
                answers[name] = QuestionAnswer(probability=ans["noul"])
            else:
                answers[name] = QuestionAnswer(probabilities=ans["probabilities"], choice=ans.get("choice"), confidence=ans.get("confidence"))
        input_tokens = (body.get("usage") or {}).get("input_tokens")
        return DecisionResponse(answers=answers, input_tokens=input_tokens, model=body.get("model"))
