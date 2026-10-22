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

# One question per rule in the documentation the agent saw on how fee rules are matched, fees computed, fraud
# measured, months bounded and domestic payments defined: manual.md lines 89, 92, 93, 95 and 225, and the compute_fees
# description (VERIFIER.md lists what is left out and why). Asked on query_data steps, where the agent writes the
# computation itself; compute_fees applies these rules on its own.
RULE_QUESTIONS = {
    "fee_formula": {
        "type": "choice",
        "instructions": "If `proposed_step.arguments` computes a fee, does it use the formula in `documentation`: "
                        "fixed_amount plus rate times the transaction value divided by 10000?",
        "criteria": {
            "yes": "The fee is fixed_amount + rate * amount / 10000, written out or through fee_amount(fixed, rate, amount)",
            "no": "It leaves out a term, divides by something other than 10000, or treats rate as a percentage",
            "not_applicable": "The step computes no fee",
        },
        "flag": "no",
    },
    "natural_months": {
        "type": "choice",
        "instructions": "If `proposed_step.arguments` computes a monthly volume or a monthly fraud level, does it "
                        "group payments by calendar month, from the first to the last day of each month?",
        "criteria": {
            "yes": "Months are calendar months, through the month column or day-of-year ranges that match them",
            "no": "It uses 30-day blocks, rolling windows, or periods that are not calendar months",
            "not_applicable": "The step computes no monthly volume or monthly fraud level",
        },
        "flag": "no",
    },
    "full_month_bands": {
        "type": "choice",
        "instructions": "If `proposed_step.arguments` computes a merchant's monthly volume or monthly fraud level "
                        "to match fee rules, does it use all of that merchant's payments in the month? The "
                        "`compute_fees` entry in `tools` says monthly bands use each full merchant month.",
        "criteria": {
            "yes": "The monthly figure covers every payment the merchant had in that calendar month",
            "no": "The monthly figure covers only some payments, such as the credit payments or the ones being priced",
            "not_applicable": "The step computes no monthly volume or fraud level for matching fee rules",
        },
        "flag": "no",
    },
    "null_matches_all": {
        "type": "choice",
        "instructions": "If `proposed_step.arguments` builds the set of fee rules that apply to a payment or "
                        "transaction, to compute, average or compare its fee, does it keep the rules whose value is "
                        "null in a field it filters on? `documentation` says a null field applies to every value.",
        "criteria": {
            "yes": "Rules with null in a filtered field are kept, for example `field = 1 OR field IS NULL`",
            "no": "A condition such as `field = 1` drops rules whose value is null, which apply to the payment too",
            "not_applicable": "The step does not build that set: a schema lookup, a sample of rows, or a look at "
                              "the rules that have explicit values",
        },
        "flag": "no",
    },
    "empty_matches_all": {
        "type": "choice",
        "instructions": "If `proposed_step.arguments` builds the set of fee rules that apply to a payment or "
                        "transaction, to compute, average or compare its fee, does it keep the rules whose list is "
                        "empty in a list field it filters on? The `compute_fees` entry in `tools` says an empty "
                        "rule field matches every value.",
        "criteria": {
            "yes": "Rules with an empty list in a filtered field are kept",
            "no": "A membership test, such as one through json_each, drops rules whose list is empty, which apply "
                  "to the payment too",
            "not_applicable": "The step does not build that set: a schema lookup, a sample of rows, or a look at "
                              "the rules that list particular values",
        },
        "flag": "no",
    },
    "sum_matching_rules": {
        "type": "choice",
        "instructions": "If `proposed_step.arguments` computes the fees of actual payments that more than one fee "
                        "rule matches, does it add up the fees of every matching rule, as the `compute_fees` entry "
                        "in `tools` says?",
        "criteria": {
            "yes": "Each payment's fee is the sum of the fees of all the rules it matches",
            "no": "It keeps one rule per payment, or averages the matching rules, or takes the cheapest or dearest",
            "not_applicable": "The step does not compute fees of actual payments",
        },
        "flag": "no",
    },
    "fraud_by_volume": {
        "type": "choice",
        "instructions": "If `question` asks about fraud and `proposed_step.arguments` computes the statistic used "
                        "to answer it, does that statistic measure fraud as fraudulent payment amount over total "
                        "payment amount, which is how `documentation` defines fraud?",
        "criteria": {
            "yes": "The statistic that ranks or reports fraud is eur_amount summed over fraudulent payments divided "
                   "by eur_amount summed over all payments; other columns computed beside it do not matter",
            "no": "The statistic that ranks or reports fraud is a count of fraudulent payments, a share of "
                  "transactions, or another measure",
            "not_applicable": "`question` is not about fraud, or the step computes no statistic used to answer it, "
                              "such as a schema lookup or a side count",
        },
        "flag": "no",
    },
    "intracountry": {
        "type": "choice",
        "instructions": "If `proposed_step.arguments` decides whether a payment is domestic, does it compare "
                        "issuing_country with acquirer_country, as `documentation` defines intracountry?",
        "criteria": {
            "yes": "A payment is domestic when issuing_country equals acquirer_country",
            "no": "It uses ip_country, the merchant's country or another field",
            "not_applicable": "The step does not decide whether payments are domestic",
        },
        "flag": "no",
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
    "cost_direction": {
        "type": "choice",
        "instructions": "Read `proposed_step.final_answer` as the answer to `question`. Does it contradict what "
                        "`documentation` says payments typically cost: more with faster capture, a higher fraud "
                        "level, credit cards or international payments, and less with a higher monthly volume?",
        "criteria": {
            "yes": "Taken with `question`, the answer says a characteristic makes fees cheaper where the "
                   "documentation says it typically makes them dearer, or the reverse",
            "no": "Taken with `question`, the answer agrees with the documentation on each such characteristic",
            "not_applicable": "The answer makes no claim about which characteristics make fees cheaper or dearer, "
                              "such as a number or a list of fee IDs",
        },
        "flag": "yes",
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
    if tool_name == "finish":
        return ANSWER_QUESTIONS
    return {**CALL_QUESTIONS, **RULE_QUESTIONS} if tool_name == "query_data" else CALL_QUESTIONS


def flag_score(question: dict, answer: dict, final_answer: str | None) -> float | None:
    """How strongly an answer flags the step, from 0 to 1, read the same way for every checker."""
    flag = question["flag"]
    if question["type"] == "noul":
        p = answer.get("noul")
        if p is None:
            return None
        if flag == "answerable_but_not_applicable":     # only a give-up can be wrongly given up
            given_up = (final_answer or "").strip().lower() in ("", "not applicable")
            return p if given_up else None
        return 1 - p
    probabilities = answer.get("probabilities") or {}
    if flag == "not_proceed":
        return 1 - probabilities.get("proceed", 0.0) if probabilities else None
    return probabilities.get(flag)
