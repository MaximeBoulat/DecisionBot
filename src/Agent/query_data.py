from __future__ import annotations

import json
import math
import sqlite3
import statistics
import time
from decimal import Decimal, localcontext, ROUND_HALF_EVEN

import pandas as pd


class Percentile:
    def __init__(self) -> None:
        self.values: list[float] = []
        self.q: float | None = None

    def step(self, value: float | None, q: float) -> None:
        if not 0 <= q <= 1 or (self.q is not None and q != self.q):
            raise ValueError("Quantile must be constant and between zero and one")
        self.q = q
        if value is not None:
            self.values.append(value)

    def finalize(self) -> float | None:
        result = None
        if self.values:
            values = sorted(self.values)
            position = (len(values) - 1) * self.q
            low = math.floor(position)
            high = math.ceil(position)
            result = values[low] + (values[high] - values[low]) * (position - low)
        return result


class StandardDeviation:
    sample = False

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def step(self, value: float | None) -> None:
        if value is not None:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (value - self.mean)

    def finalize(self) -> float | None:
        denominator = self.count - int(self.sample)
        result = math.sqrt(max(0, self.m2 / denominator)) if denominator > 0 else None
        return result


class SampleDeviation(StandardDeviation):
    sample = True


class Correlation:
    def __init__(self) -> None:
        self.x: list[float] = []
        self.y: list[float] = []

    def step(self, x: float | None, y: float | None) -> None:
        if x is not None and y is not None:
            self.x.append(x)
            self.y.append(y)

    def finalize(self) -> float | None:
        result = None
        if len(self.x) > 1 and max(self.x) != min(self.x) and max(self.y) != min(self.y):
            result = statistics.correlation(self.x, self.y)
        return result


class DecimalSum:
    average = False

    def __init__(self) -> None:
        self.total = Decimal(0)
        self.count = 0

    def step(self, value: str | float | None) -> None:
        if value is not None:
            with localcontext() as context:
                context.prec = 60
                self.total += Decimal(str(value))
            self.count += 1

    def finalize(self) -> str | None:
        result = None
        if self.count:
            with localcontext() as context:
                context.prec = 60
                result = str(self.total / self.count if self.average else self.total)
        return result


class DecimalAverage(DecimalSum):
    average = True


def fee_amount(fixed: float, rate: float, amount: float) -> str:
    with localcontext() as context:
        context.prec = 60
        result = str(Decimal(str(fixed)) + Decimal(str(rate)) * Decimal(str(amount)) / 10000)
    return result


def decimal_round(value: str | None, places: int) -> str | None:
    result = None
    if value is not None:
        with localcontext() as context:
            context.prec = 60
            result = format(Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN), f".{places}f")
    return result


class DataQuery:
    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.columns = {name: list(table.columns) for name, table in tables.items()}
        for name, table in tables.items():
            serializable = table.copy()
            for column in serializable:
                serializable[column] = serializable[column].map(
                    lambda value: json.dumps(value) if isinstance(value, (list, dict)) else value
                )
            serializable.to_sql(name, self.connection, index=False)
        self.connection.create_aggregate("percentile", 2, Percentile)
        self.connection.create_aggregate("stddev_pop", 1, StandardDeviation)
        self.connection.create_aggregate("stddev_samp", 1, SampleDeviation)
        self.connection.create_aggregate("corr", 2, Correlation)
        self.connection.create_aggregate("decimal_sum", 1, DecimalSum)
        self.connection.create_aggregate("decimal_avg", 1, DecimalAverage)
        self.connection.create_function("fee_amount", 3, fee_amount)
        self.connection.create_function("decimal_round", 2, decimal_round)
        self.connection.execute("SELECT value FROM json_each('[]')").fetchall()
        self.connection.set_authorizer(self._authorize)

    def _authorize(self, action: int, arg1: str | None, arg2: str | None, database: str | None, source: str | None) -> int:
        allowed = action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE)
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in ("load_extension", "writefile", "readfile"):
            allowed = False
        return sqlite3.SQLITE_OK if allowed else sqlite3.SQLITE_DENY

    def query(self, sql: str | None, offset: int, limit: int) -> dict:
        if offset < 0 or not 1 <= limit <= 500:
            raise ValueError("offset must be nonnegative and limit must be between 1 and 500")
        if sql is None:
            result = {
                "tables": self.columns,
                "functions": ["percentile(value,q)", "stddev_pop(value)", "stddev_samp(value)", "corr(x,y)",
                              "fee_amount(fixed,rate,amount)", "decimal_sum(value)", "decimal_avg(value)",
                              "decimal_round(value,places)"],
                "notes": "Use ORDER BY for stable pagination. NULL values are excluded by aggregates. "
                         "Percentile uses linear interpolation. Decimal rounding uses half-even. "
                         "Fee and merchant list columns are JSON arrays; empty-list matching must be chosen explicitly.",
            }
        else:
            deadline = time.monotonic() + 30
            self.connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
            try:
                cursor = self.connection.execute(sql)
                if cursor.description is None:
                    raise ValueError("Only result-producing queries are supported")
                columns = [column[0] for column in cursor.description]
                if len(set(columns)) != len(columns):
                    raise ValueError("Use unique column aliases")
                rows: list[dict] = []
                total = 0
                for values in cursor:
                    if offset <= total < offset + limit:
                        rows.append(dict(zip(columns, values)))
                    total += 1
                result = {
                    "rows": rows, "total_rows": total, "returned_rows": len(rows), "offset": offset,
                    "truncated": offset > 0 or offset + len(rows) < total,
                    "next_offset": offset + len(rows) if offset + len(rows) < total else None,
                }
            finally:
                self.connection.set_progress_handler(None, 0)
        return result
