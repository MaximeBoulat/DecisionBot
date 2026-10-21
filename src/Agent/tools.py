from __future__ import annotations

import calendar
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd

from Agent.fee_calculator import MANUAL_POLICY, FeeCalculator, FeeScenario
from Agent.query_data import DataQuery
from Agent.tool_schemas import TOOLS_SCHEMA


def _month_day_range(year: int, month: int) -> tuple[int, int]:
    start = date(year, month, 1).timetuple().tm_yday
    end = date(year, month, calendar.monthrange(year, month)[1]).timetuple().tm_yday
    return start, end


class ToolsLayer:
    def __init__(self, data_dir: Path) -> None:
        context = data_dir / "context"
        self.payments = pd.read_csv(context / "payments.csv")
        self.payments["month"] = (
            pd.to_datetime(self.payments.year.astype(str), format="%Y")
            + pd.to_timedelta(self.payments.day_of_year - 1, unit="D")
        ).dt.month
        self.payments["amount_cents"] = self.payments.eur_amount.map(lambda value: int(Decimal(str(value)) * 100))
        self.fees = json.loads((context / "fees.json").read_text())
        merchant_rows = json.loads((context / "merchant_data.json").read_text())
        self.merchants = {row["merchant"]: row for row in merchant_rows}
        self.fee_calculator = FeeCalculator(self.payments, self.fees, self.merchants)
        self.data_query = DataQuery({
            "payments": self.payments,
            "fees": pd.DataFrame(self.fees),
            "merchants": pd.DataFrame(merchant_rows),
            "merchant_category_codes": pd.read_csv(context / "merchant_category_codes.csv").drop(columns=["Unnamed: 0"], errors="ignore"),
            "acquirer_countries": pd.read_csv(context / "acquirer_countries.csv").drop(columns=["Unnamed: 0"], errors="ignore"),
        })

    def dispatch(self, name: str, args: dict) -> str:
        try:
            if name == "query_transactions":
                result = self._query_transactions(**args)
            elif name == "compute_fees":
                result = self._compute_fees(**args)
            elif name == "query_data":
                result = self.data_query.query(**args)
            else:
                raise ValueError(f"unknown tool {name!r}")
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result, allow_nan=False)

    def _filtered(self, merchant: str | None, year: int | None, month: int | None,
                  day_of_year_min: int | None, day_of_year_max: int | None,
                  card_scheme: str | None = None, is_credit: bool | None = None,
                  aci: str | None = None, exclude_refused: bool = False,
                  exclude_fraudulent: bool = False) -> pd.DataFrame:
        df = self.payments
        if merchant is not None:
            if merchant not in self.merchants:
                raise ValueError(f"Unknown merchant {merchant}")
            df = df[df.merchant == merchant]
        if year is not None:
            df = df[df.year == year]
        if month is not None:
            if not 1 <= month <= 12:
                raise ValueError("month must be between 1 and 12")
            df = df[df.month == month]
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

    def _query_transactions(self, merchant: str | None, year: int | None, month: int | None,
                            day_of_year_min: int | None, day_of_year_max: int | None,
                            card_scheme: str | None, is_credit: bool | None, aci: str | None,
                            exclude_refused: bool, exclude_fraudulent: bool,
                            group_by: list[str] | None, aggregate: dict | None,
                            offset: int = 0, limit: int = 300) -> dict:
        if offset < 0 or not 1 <= limit <= 500:
            raise ValueError("Invalid pagination")
        df = self._filtered(merchant, year, month, day_of_year_min, day_of_year_max, card_scheme,
                            is_credit, aci, exclude_refused, exclude_fraudulent)
        source_count = len(df)
        if aggregate is not None:
            column, function = aggregate["column"], aggregate["function"]
            if function not in ("sum", "count", "size", "mean", "min", "max", "nunique", "std", "quantile"):
                raise ValueError("Unsupported aggregation")
            target = df.groupby(group_by, dropna=False)[column] if group_by else df[column]
            if function == "quantile":
                q = aggregate["quantile"]
                if q is None or not 0 <= q <= 1:
                    raise ValueError("quantile requires a fraction between zero and one")
                value = target.quantile(q)
            elif function == "size" and not group_by:
                value = target.size
            else:
                value = getattr(target, function)()
            result = value.reset_index(name="value") if group_by else pd.DataFrame({"value": [value]})
        elif group_by:
            result = df.groupby(group_by, dropna=False).size().reset_index(name="count")
        else:
            result = df
        rows = json.loads(result.iloc[offset:offset + limit].to_json(orient="records", double_precision=15))
        return {
            "rows": rows, "source_payment_count": source_count, "total_rows": len(result),
            "returned_rows": len(rows), "offset": offset,
            "truncated": offset > 0 or offset + len(rows) < len(result),
            "next_offset": offset + len(rows) if offset + len(rows) < len(result) else None,
        }

    def _compute_fees(self, merchant: str | None, year: int, month: int | None,
                      day_of_year_min: int | None, day_of_year_max: int | None,
                      card_scheme: str | None, exclude_refused: bool,
                      scenario: dict | None, offset: int = 0, limit: int = 300) -> dict:
        if offset < 0 or not 1 <= limit <= 500:
            raise ValueError("Invalid pagination")
        selected = self._filtered(merchant, year, month, day_of_year_min, day_of_year_max,
                                  card_scheme=card_scheme, exclude_refused=exclude_refused)
        return self.fee_calculator.compute(
            selected, MANUAL_POLICY, FeeScenario(**scenario) if scenario is not None else None,
            offset, limit,
        )
