from __future__ import annotations

import hashlib
import json
import math
import string
import time
from pathlib import Path

from Agent.retry import post_json
from Verifier.questions import qhash, wire

# A correct decision-model server puts billing near 0.98 (llama.cpp Metal fusion bug otherwise).
CANARY_STATE = "I was charged twice for my subscription this month and need a refund for the duplicate payment."
CANARY_QUESTIONS = {"department": {"type": "choice", "instructions": "Which team should handle the message?",
                                   "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}}}
REVIEWER_SYSTEM = ("You check one step of a data-analysis agent before it takes effect. You are given the agent's "
                   "documentation, the question it is answering, its earlier steps, the step it proposes, and facts "
                   "computed by code. Answer the question about that step with the letter of one option.")


def state_hash(state: dict) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]


def invalid(answers: dict, questions: dict) -> str | None:
    ok = lambda v: isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1
    for name, q in questions.items():
        answer = answers.get(name) if isinstance(answers, dict) else None
        if not isinstance(answer, dict):
            return f"no answer for {name}"
        if q["type"] == "noul" and not ok(answer.get("noul")):
            return f"bad noul for {name}"
        if q["type"] == "choice":
            p = answer.get("probabilities") or {}
            if set(p) != set(q["criteria"]) or not all(ok(v) for v in p.values()):
                return f"bad probabilities for {name}"
    return None


class DecisionModelChecker:
    """A decision model behind a /v1/systemone endpoint; all questions go in one request."""

    def __init__(self, model: str, url: str, timeout: float = 300, api_key: str | None = None) -> None:
        self.model, self.url, self.timeout = model, url, timeout
        self.headers = {"content-type": "application/json", **({"authorization": f"Bearer {api_key}"} if api_key else {})}
        self.name = f"decision:{model}"

    def canary(self) -> str:
        body = post_json(self.url, {"model": self.model, "state": CANARY_STATE, "questions": CANARY_QUESTIONS},
                         self.headers, self.timeout)
        p = body["answers"]["department"]["probabilities"]["billing"]
        if p < 0.9:
            raise RuntimeError(f"canary failed: billing = {p:.3f}, expected about 0.98")
        return f"billing = {p:.3f}"

    def ask(self, state: dict, questions: dict) -> tuple[dict, int | None]:
        body = post_json(self.url, {"model": self.model, "state": state, "questions": wire(questions)},
                         self.headers, self.timeout)
        return body["answers"], (body.get("usage") or {}).get("input_tokens")


class ReviewerChecker:
    """An LLM behind an OpenAI-compatible chat endpoint. Each question is one single-token reply whose
    option probabilities come from the token log-probabilities, so it is scored like a decision model.
    Thinking is off: the reply is the first token."""

    def __init__(self, model: str, url: str, timeout: float = 300) -> None:
        self.model, self.url, self.timeout = model, url, timeout
        self.headers = {"content-type": "application/json"}
        self.name = f"reviewer:{model}"

    def _options(self, q: dict) -> dict[str, str]:
        criteria = q["criteria"] if q["type"] == "choice" else {"yes": "Yes", "no": "No"}
        return dict(zip(string.ascii_uppercase, criteria))

    def _probabilities(self, state_text: str, q: dict) -> tuple[dict[str, float], int | None]:
        letters = self._options(q)
        criteria = q["criteria"] if q["type"] == "choice" else {"yes": "Yes", "no": "No"}
        options = "\n".join(f"{letter}. {name}: {criteria[name]}" for letter, name in letters.items())
        body = post_json(self.url, {
            "model": self.model, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": REVIEWER_SYSTEM},
                         {"role": "user", "content": f"{state_text}\n\nQUESTION: {q['instructions']}\n"
                                                     f"OPTIONS:\n{options}\nReply with one letter."}],
        }, self.headers, self.timeout)
        top = body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        mass = {}
        for item in top:
            letter = item["token"].strip().upper()
            if letter in letters:
                mass[letters[letter]] = mass.get(letters[letter], 0.0) + math.exp(item["logprob"])
        total = sum(mass.values())
        if total <= 0:
            raise ValueError(f"no option letter among the top tokens: {[t['token'] for t in top]}")
        return {name: mass.get(name, 0.0) / total for name in criteria}, (body.get("usage") or {}).get("prompt_tokens")

    def canary(self) -> str:
        q = {"type": "choice", "instructions": "Which team should handle `message`?",
             "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}}
        p, _ = self._probabilities(json.dumps({"message": CANARY_STATE}), q)
        if p["billing"] < 0.9:
            raise RuntimeError(f"canary failed: billing = {p['billing']:.3f}")
        return f"billing = {p['billing']:.3f}"

    def ask(self, state: dict, questions: dict) -> tuple[dict, int | None]:
        state_text = "STATE:\n" + json.dumps(state, ensure_ascii=False)
        answers, tokens = {}, None
        for name, q in wire(questions).items():
            p, tokens = self._probabilities(state_text, q)
            answers[name] = {"noul": p["yes"]} if q["type"] == "noul" else \
                {"probabilities": p, "choice": max(p, key=p.get), "confidence": max(p.values())}
        return answers, tokens


class VerdictCache:
    """Raw answers keyed by point, checker, question set and exact state; reruns make no requests."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[tuple, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if "answers" in row:
                    self.rows[(row["point_id"], row["checker"], row["qhash"], row["state_hash"])] = row

    def get(self, point_id: str, checker: str, questions: dict, state: dict) -> dict | None:
        return self.rows.get((point_id, checker, qhash(questions), state_hash(state)))

    def ask(self, checker, point_id: str, state: dict, questions: dict) -> dict:
        key = (point_id, checker.name, qhash(questions), state_hash(state))
        row = {"point_id": point_id, "checker": checker.name, "qhash": key[2], "state_hash": key[3]}
        started = time.perf_counter()
        try:
            answers, tokens = checker.ask(state, questions)
            problem = invalid(answers, wire(questions))
            row.update({"error": problem} if problem else {"answers": answers, "input_tokens": tokens})
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["latency_s"] = round(time.perf_counter() - started, 3)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if "answers" in row:
            self.rows[key] = row
        return row
