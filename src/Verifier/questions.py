from __future__ import annotations

import hashlib
import json

# One property per question; `facts` holds date readings and answer locations computed in code.
# Each question names the option that flags the step, so code can score every checker the same way.
ACTION = {
    "proceed": "The step fits the question and the documentation as written",
    "revise": "Something in the step is wrong: the period, merchant, filter, change or reported quantity",
    "recheck": "The step relies on a rule the agent has not checked in `documentation` or earlier results",
    "ask": "The question allows more than one reading and the user must choose",
}

CALL_QUESTIONS = {
    "period": {
        "type": "choice",
        "instructions": "Does `proposed_step.arguments` select the time period that `question` asks about? "
                        "`facts` gives the calendar dates of any day or month arguments.",
        "criteria": {
            "yes": "The arguments select exactly the period the question names, or all of it when it names none",
            "no": "The arguments select a different, shorter or longer period than the question names",
            "not_applicable": "The step does not select payments by date, such as a schema or fee-rule lookup",
        },
        "flag": "no",
    },
    "entity": {
        "type": "choice",
        "instructions": "Does `proposed_step.arguments` select the merchant or merchants that `question` asks about?",
        "criteria": {
            "yes": "The arguments select the merchant the question names, or all merchants when it names none",
            "no": "The arguments select a different merchant, or all merchants when the question names one",
            "not_applicable": "The step does not select payments by merchant, such as a schema or fee-rule lookup",
        },
        "flag": "no",
    },
    "filter": {
        "type": "choice",
        "instructions": "Does `proposed_step.arguments` drop records that neither `question` nor `documentation` "
                        "says to drop?",
        "criteria": {
            "yes": "It excludes records, such as refused or fraudulent payments, that nothing asks to exclude",
            "no": "Every exclusion is asked for by the question or the documentation, or nothing is excluded",
            "not_applicable": "The step does not select records",
        },
        "flag": "yes",
    },
    "change": {
        "type": "choice",
        "instructions": "Does `proposed_step.arguments` change only what `question` says to change?",
        "criteria": {
            "yes": "The hypothetical change matches the question: the same fee rule, value, code or account type",
            "no": "It changes something the question does not mention, or a different rule or value",
            "not_applicable": "The question asks about actual payments, or the step makes no hypothetical change",
        },
        "flag": "no",
    },
    "action": {
        "type": "choice",
        "instructions": "What should happen to `proposed_step` before it runs?",
        "criteria": ACTION,
        "flag": "not_proceed",
    },
}

ANSWER_QUESTIONS = {
    "quantity": {
        "type": "choice",
        "instructions": "Does `proposed_step.final_answer` report the quantity that `question` asks for? "
                        "`facts` says which tool result each number came from.",
        "criteria": {
            "yes": "It reports what was asked: a change when a change is asked, a total when a total is asked",
            "no": "It reports a related but different quantity, such as a total instead of a change",
            "not_observable": "The final answer is 'Not Applicable' or empty, so it reports no quantity",
        },
        "flag": "no",
    },
    "supported": {
        "type": "choice",
        "instructions": "Do the tool results in `history` support `proposed_step.final_answer`?",
        "criteria": {
            "yes": "A tool result contains the answer or the values it is directly computed from",
            "no": "No tool result supports it, or a result contradicts it",
            "not_observable": "The relevant results are omitted from `history`",
        },
        "flag": "no",
    },
    "answerable": {
        "type": "noul",
        "instructions": "Do the tool results in `history` already contain an answer to `question`?",
        "flag": "answerable_but_not_applicable",
    },
    "format": {
        "type": "noul",
        "instructions": "Does `proposed_step.final_answer` follow `answer_format`, including its rounding?",
        "flag": "no",
    },
    "action": {
        "type": "choice",
        "instructions": "What should happen to `proposed_step` before the answer is returned?",
        "criteria": ACTION,
        "flag": "not_proceed",
    },
}


def wire(questions: dict) -> dict:
    """The questions as sent: without the scoring key, which the model never sees."""
    return {name: {k: v for k, v in q.items() if k != "flag"} for name, q in questions.items()}


def qhash(questions: dict) -> str:
    return hashlib.sha256(json.dumps(wire(questions), sort_keys=True).encode()).hexdigest()[:12]


def questions_for(tool_name: str) -> dict:
    return ANSWER_QUESTIONS if tool_name == "finish" else CALL_QUESTIONS


def flag_score(question: dict, answer: dict, final_answer: str | None) -> float | None:
    """How strongly an answer flags the step, from 0 to 1, read the same way for every checker."""
    flag = question["flag"]
    if question["type"] == "noul":
        p = answer.get("noul")
        if p is None:
            return None
        if flag == "answerable_but_not_applicable":
            given_up = (final_answer or "").strip().lower() in ("", "not applicable")
            return p if given_up else 0.0
        return 1 - p
    probabilities = answer.get("probabilities") or {}
    if flag == "not_proceed":
        return 1 - probabilities.get("proceed", 0.0) if probabilities else None
    return probabilities.get(flag)
