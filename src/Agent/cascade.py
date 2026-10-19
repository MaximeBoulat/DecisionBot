from __future__ import annotations

import math
import time
from dataclasses import asdict

from Agent.context import DecisionContext
from Agent.decision_model import DecisionModel
from debug_log import DebugLog
from models import DecisionQuestion, DecisionRequest, DecisionRecord


class Cascade:
    def __init__(self, model: DecisionModel, tool_threshold: float | None, argument_threshold: float | None) -> None:
        for threshold in (tool_threshold, argument_threshold):
            if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
                raise ValueError("Thresholds must be between zero and one, or null to always defer")
        self.model = model
        self.thresholds = {"tool": tool_threshold, "argument": argument_threshold}

    def decide(self, context: DecisionContext, point: str, question: DecisionQuestion,
               debug_log: DebugLog, shadow: bool = False) -> DecisionRecord:
        threshold = self.thresholds[point]
        debug_log.write("jev_request", point=point, state=context.state, question=asdict(question))
        print(f"=======> Asking Jev for {point}...", end="", flush=True)
        started = time.perf_counter()
        try:
            response = self.model.invoke(context, DecisionRequest((question,)))
            answer = response.answers[question.name]
            probabilities = answer.probabilities
            if not probabilities or set(probabilities) != set(question.criteria or {}):
                raise ValueError("Jev must return a probability for every supplied option")
            if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
                raise ValueError("Invalid choice probability")
            if not math.isclose(sum(probabilities.values()), 1, abs_tol=0.02):
                raise ValueError("Choice probabilities do not sum to one")
            choice = max(probabilities, key=probabilities.get)
            confidence = probabilities[choice]
            accepted = not shadow and choice != "other" and threshold is not None and confidence >= threshold
            record = DecisionRecord(point, question, choice, confidence, threshold, accepted,
                                    response=response)
            status = "Shadow" if shadow else ("Accepted" if accepted else "Deferred")
            print(f" {status} — probability {confidence:.3f}.", flush=True)
        except Exception as exc:
            record = DecisionRecord(point, question, None, None, threshold, False,
                                    error=f"{type(exc).__name__}: {exc}")
            print(" Unavailable — deferring to generator.", flush=True)
        debug_log.write("jev_decision", record=asdict(record), elapsed_seconds=time.perf_counter() - started)
        return record
