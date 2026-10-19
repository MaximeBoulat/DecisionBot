from __future__ import annotations


def object_schema(properties: dict) -> dict:
    return {
        "type": "object", "properties": properties,
        "required": list(properties), "additionalProperties": False,
    }


def nullable(kind: str, description: str = "") -> dict:
    return {"type": [kind, "null"], "description": description}


def tool(name: str, description: str, properties: dict) -> dict:
    return {
        "type": "function", "name": name, "description": description,
        "parameters": object_schema(properties), "strict": True,
    }


AGGREGATE_SCHEMA = object_schema({
    "column": {"type": "string"},
    "function": {"type": "string", "enum": ["sum", "count", "size", "mean", "min", "max", "nunique", "std", "quantile"]},
    "quantile": nullable("number", "Required only for quantile: fraction from 0 to 1."),
})
AGGREGATE_SCHEMA["type"] = ["object", "null"]

POLICY_SCHEMA = object_schema({
    "empty_lists": {"type": "string", "enum": ["wildcard", "match_nothing"]},
    "overlap": {"type": "string", "enum": ["sum", "mean", "min", "max", "error"]},
    "unmatched": {"type": "string", "enum": ["unknown", "zero", "error"]},
    "immediate_capture": {"type": "string", "enum": ["literal", "zero_days"]},
})

SCENARIO_SCHEMA = object_schema({
    "merchant_category_code": nullable("integer"),
    "account_type": nullable("string", "Replace the merchant's account type."),
    "card_scheme": nullable("string", "Replace scheme on selected payments, rather than filtering."),
    "aci": nullable("string", "Replace ACI on selected payments."),
    "fraudulent_only": {"type": "boolean", "description": "Apply scheme/ACI changes only to fraudulent payments."},
    "fee_rule_id": nullable("integer", "Rule to change; required when changing its rate or account list."),
    "rate": nullable("number", "Replacement rate for the named fee rule."),
    "rule_account_type": nullable("string", "Replace the named rule's account list with this singleton."),
})
SCENARIO_SCHEMA["type"] = ["object", "null"]

TOOLS_SCHEMA = [
    tool("query_transactions",
         "Filter payments across all merchants or one merchant; aggregate over ALL filtered rows before pagination. "
         "Results include total_rows and next_offset. Month and day bounds intersect. For joins, percentiles, "
         "correlations, repeat shoppers or arbitrary filters use query_data.",
         {
             "merchant": nullable("string", "Null selects all merchants."),
             "year": nullable("integer", "Null selects all years."),
             "month": nullable("integer", "1-12; intersects explicit day bounds."),
             "day_of_year_min": nullable("integer"),
             "day_of_year_max": nullable("integer"),
             "card_scheme": nullable("string"),
             "is_credit": nullable("boolean"),
             "aci": nullable("string"),
             "exclude_refused": {"type": "boolean"},
             "exclude_fraudulent": {"type": "boolean"},
             "group_by": {"type": ["array", "null"], "items": {"type": "string"}},
             "aggregate": AGGREGATE_SCHEMA,
             "offset": {"type": "integer", "minimum": 0},
             "limit": {"type": "integer", "minimum": 1, "maximum": 500},
         }),
    tool("query_data",
         "Execute one read-only SQLite SELECT or WITH query against payments, fees, merchants, "
         "merchant_category_codes, acquirer_countries. sql=null describes columns and functions. "
         "Lists in fees/merchants are JSON: use json_each for membership. Booleans are 0/1. "
         "Supports COUNT(DISTINCT), window functions, percentile(value,q), stddev_pop(value), "
         "stddev_samp(value), corr(x,y), fee_amount(fixed,rate,amount), decimal_sum(value), "
         "decimal_avg(value), decimal_round(value,places). Decimal functions return exact decimal strings. "
         "payments also has month and amount_cents. No answer key or agent submissions are available.",
         {
             "sql": nullable("string"),
             "offset": {"type": "integer", "minimum": 0},
             "limit": {"type": "integer", "minimum": 1, "maximum": 500},
         }),
    tool("compute_fees",
         "Compute observed-payment fees with explicit interpretation policy. Null merchant/month selects all "
         "merchants/the full year; monthly bands always use each original full merchant/calendar month. "
         "Returns baseline, scenario and new-minus-old delta using decimal arithmetic. "
         "Policy choices are assumptions: the manual does not define empty lists, rule overlap or unmatched "
         "prices. unmatched=unknown keeps total null if any payment lacks a rule. "
         "Provides coverage counts, matched IDs and affected merchants. All summaries cover the full selection; "
         "only transaction details are paginated. Scenario changes do not predict altered fraud incidence.",
         {
             "merchant": nullable("string"),
             "year": {"type": "integer"},
             "month": nullable("integer"),
             "day_of_year_min": nullable("integer"),
             "day_of_year_max": nullable("integer"),
             "card_scheme": nullable("string", "Filter original payments; scenario.card_scheme changes them."),
             "exclude_refused": {"type": "boolean"},
             "policy": POLICY_SCHEMA,
             "scenario": SCENARIO_SCHEMA,
             "offset": {"type": "integer", "minimum": 0},
             "limit": {"type": "integer", "minimum": 1, "maximum": 500},
         }),
]
