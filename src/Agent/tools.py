from __future__ import annotations

import calendar
import json
from datetime import date
from pathlib import Path

import pandas as pd

TOOLS_SCHEMA = [
    {
        "type": "function",
        "name": "query_transactions",
        "description": "Filter and optionally group/aggregate payment transactions for one merchant.",
        "parameters": {
            "type": "object",
            "properties": {
                "merchant": {"type": "string", "description": "Exact merchant name."},
                "year": {"type": "integer"},
                "month": {"type": ["integer", "null"],
                          "description": "1-12, the natural calendar month. Ignored if day_of_year_min/max given."},
                "day_of_year_min": {"type": ["integer", "null"]},
                "day_of_year_max": {"type": ["integer", "null"]},
                "card_scheme": {"type": ["string", "null"]},
                "is_credit": {"type": ["boolean", "null"]},
                "aci": {"type": ["string", "null"]},
                "exclude_refused": {"type": "boolean"},
                "exclude_fraudulent": {"type": "boolean"},
                "group_by": {"type": ["array", "null"], "items": {"type": "string"}},
                "aggregate": {
                    "type": ["object", "null"],
                    "description": "One aggregation applied after group_by.",
                    "properties": {
                        "column": {"type": "string"},
                        "function": {"type": "string", "enum": ["sum", "count", "mean", "min", "max"]},
                    },
                    "required": ["column", "function"],
                    "additionalProperties": False,
                },
            },
            "required": ["merchant", "year", "month", "day_of_year_min", "day_of_year_max", "card_scheme",
                         "is_credit", "aci", "exclude_refused", "exclude_fraudulent", "group_by", "aggregate"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "compute_fees",
        "description": "Compute processing fees per the merchant guide's fee rules for a merchant's transactions.",
        "parameters": {
            "type": "object",
            "properties": {
                "merchant": {"type": "string"},
                "year": {"type": "integer"},
                "month": {"type": "integer",
                          "description": "1-12, the natural month defining monthly volume and fraud rate."},
                "day_of_year_min": {"type": ["integer", "null"],
                                    "description": "Restrict which transactions get a fee; default the whole month."},
                "day_of_year_max": {"type": ["integer", "null"]},
                "card_scheme": {"type": ["string", "null"], "description": "Restrict to one card scheme."},
                "exclude_refused": {"type": "boolean"},
            },
            "required": ["merchant", "year", "month", "day_of_year_min", "day_of_year_max", "card_scheme",
                         "exclude_refused"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def _month_day_range(year: int, month: int) -> tuple[int, int]:
    start = date(year, month, 1).timetuple().tm_yday
    last_day = calendar.monthrange(year, month)[1]
    end = date(year, month, last_day).timetuple().tm_yday
    return start, end


def _parse_bound(token: str) -> float:
    token = token.strip()
    if token.endswith("%"):
        return float(token[:-1]) / 100
    if token.endswith("k"):
        return float(token[:-1]) * 1e3
    if token.endswith("m"):
        return float(token[:-1]) * 1e6
    return float(token)


def _in_range(value: float, spec: str | None) -> bool:
    if spec is None:
        return True
    if spec.startswith("<"):
        return value < _parse_bound(spec[1:])
    if spec.startswith(">"):
        return value > _parse_bound(spec[1:])
    lo, hi = spec.split("-", 1)
    return _parse_bound(lo) <= value <= _parse_bound(hi)


def _capture_delay_matches(merchant_value: str, rule_value: str | None) -> bool:
    if rule_value is None:
        return True
    if rule_value in ("immediate", "manual"):
        return merchant_value == rule_value
    if merchant_value in ("immediate", "manual"):
        return False
    days = float(merchant_value)
    if rule_value.startswith("<"):
        return days < float(rule_value[1:])
    if rule_value.startswith(">"):
        return days > float(rule_value[1:])
    lo, hi = rule_value.split("-", 1)
    return float(lo) <= days <= float(hi)


class ToolsLayer:
    def __init__(self, data_dir: Path) -> None:
        self.payments = pd.read_csv(data_dir / "context" / "payments.csv")
        self.fees = json.loads((data_dir / "context" / "fees.json").read_text())
        self.merchants = {
            m["merchant"]: m for m in json.loads((data_dir / "context" / "merchant_data.json").read_text())
        }

    def dispatch(self, name: str, args: dict) -> str:
        try:
            if name == "query_transactions":
                return self._query_transactions(**args)
            if name == "compute_fees":
                return self._compute_fees(**args)
            return json.dumps({"error": f"unknown tool {name!r}"})
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def _filtered(self, merchant, year, month, day_of_year_min, day_of_year_max, card_scheme=None,
                  is_credit=None, aci=None, exclude_refused=False, exclude_fraudulent=False) -> pd.DataFrame:
        df = self.payments[(self.payments.merchant == merchant) & (self.payments.year == year)]
        if month is not None:
            start, end = _month_day_range(year, month)
            df = df[(df.day_of_year >= start) & (df.day_of_year <= end)]
        if day_of_year_min is not None:
            df = df[df.day_of_year >= day_of_year_min]
        if day_of_year_max is not None:
            df = df[df.day_of_year <= day_of_year_max]
        if card_scheme is not None:
            df = df[df.card_scheme == card_scheme]
        if is_credit is not None:
            df = df[df.is_credit == is_credit]
        if aci is not None:
            df = df[df.aci == aci]
        if exclude_refused:
            df = df[~df.is_refused_by_adyen]
        if exclude_fraudulent:
            df = df[~df.has_fraudulent_dispute]
        return df

    def _query_transactions(self, merchant, year, month, day_of_year_min, day_of_year_max, card_scheme,
                             is_credit, aci, exclude_refused, exclude_fraudulent, group_by, aggregate) -> str:
        df = self._filtered(merchant, year, month, day_of_year_min, day_of_year_max, card_scheme, is_credit,
                             aci, exclude_refused, exclude_fraudulent)
        if group_by:
            grouped = df.groupby(group_by)
            if aggregate:
                result = grouped[aggregate["column"]].agg(aggregate["function"]).reset_index()
            else:
                result = grouped.size().reset_index(name="count")
            return result.to_json(orient="records")
        cols = ["psp_reference", "card_scheme", "is_credit", "eur_amount", "aci", "issuing_country",
                "acquirer_country", "is_refused_by_adyen", "has_fraudulent_dispute", "day_of_year"]
        return df[cols].head(300).to_json(orient="records")

    def _compute_fees(self, merchant, year, month, day_of_year_min, day_of_year_max, card_scheme,
                       exclude_refused) -> str:
        profile = self.merchants[merchant]
        monthly = self._filtered(merchant, year, month, None, None)
        volume = float(monthly.eur_amount.sum())
        fraud_rate = float(monthly.loc[monthly.has_fraudulent_dispute, "eur_amount"].sum() / volume) if volume else 0.0

        df = monthly
        if day_of_year_min is not None:
            df = df[df.day_of_year >= day_of_year_min]
        if day_of_year_max is not None:
            df = df[df.day_of_year <= day_of_year_max]
        if card_scheme is not None:
            df = df[df.card_scheme == card_scheme]
        if exclude_refused:
            df = df[~df.is_refused_by_adyen]

        rules = [
            r for r in self.fees
            if (not r["merchant_category_code"] or profile["merchant_category_code"] in r["merchant_category_code"])
            and (not r["account_type"] or profile["account_type"] in r["account_type"])
            and _capture_delay_matches(profile["capture_delay"], r["capture_delay"])
            and _in_range(volume, r["monthly_volume"])
            and _in_range(fraud_rate, r["monthly_fraud_level"])
        ]

        breakdown = []
        total = 0.0
        for _, t in df.iterrows():
            matches = [
                r for r in rules
                if r["card_scheme"] == t.card_scheme
                and (r["is_credit"] is None or r["is_credit"] == t.is_credit)
                and (not r["aci"] or t.aci in r["aci"])
                and (r["intracountry"] is None or r["intracountry"] == (t.issuing_country == t.acquirer_country))
            ]
            fee = sum(r["fixed_amount"] + r["rate"] * t.eur_amount / 10000 for r in matches)
            total += fee
            breakdown.append({"psp_reference": int(t.psp_reference), "rule_ids": [r["ID"] for r in matches],
                               "fee": round(fee, 6)})
        return json.dumps({
            "monthly_volume": round(volume, 2),
            "monthly_fraud_rate": round(fraud_rate, 6),
            "transactions": breakdown,
            "total_fee": round(total, 6),
        })
