from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def fit(rows: list[dict], point: str) -> dict:
    eligible = []
    for row in rows:
        if row.get("point") != point or row.get("event") != "shadow_pair":
            continue
        decision = row["decision"]
        if decision["error"] or decision["choice"] == "other":
            continue
        if type(row.get("jev_correct")) is not bool or type(row.get("llm_correct")) is not bool:
            raise ValueError("Shadow pairs require independent jev_correct and llm_correct labels")
        cost, value = row.get("saved_cost"), row.get("error_value")
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
            raise ValueError("saved_cost must be a measured nonnegative incremental LLM cost")
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("error_value must be a positive measured loss in the same units as saved_cost")
        eligible.append(row)
    eligible.sort(key=lambda row: row["decision"]["confidence"])
    if len(eligible) < 20:
        raise ValueError(f"{point}: at least 20 labeled decisions are required")
    count = len(eligible) // 20
    bins = [eligible[i * len(eligible) // count:(i + 1) * len(eligible) // count] for i in range(count)]
    merged = []
    for batch in bins:
        if merged and merged[-1][-1]["decision"]["confidence"] == batch[0]["decision"]["confidence"]:
            merged[-1].extend(batch)
        else:
            merged.append(batch)
    reports = []
    for batch in merged:
        advantage = sum(int(row["jev_correct"]) - int(row["llm_correct"])
                        + row["saved_cost"] / row["error_value"] for row in batch) / len(batch)
        reports.append({"minimum_confidence": batch[0]["decision"]["confidence"],
                        "maximum_confidence": batch[-1]["decision"]["confidence"],
                        "count": len(batch), "advantage": advantage})
    threshold = None
    for batch in reversed(reports):
        if batch["advantage"] < 0:
            break
        threshold = batch["minimum_confidence"]
    return {"threshold": threshold, "bins": reports, "count": len(eligible)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labeled_pairs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--jev-model", required=True)
    options = parser.parse_args()
    rows = [json.loads(line) for line in options.labeled_pairs.read_text().splitlines() if line.strip()]
    fits = {point: fit(rows, point) for point in ("tool", "argument")}
    profile = {
        "generator_model": options.generator_model, "jev_model": options.jev_model,
        "confidence_statistic": "max_option_probability",
        "tool_threshold": fits["tool"]["threshold"],
        "argument_threshold": fits["argument"]["threshold"],
        "calibration_task_ids": sorted({row["task_id"] for row in rows if row.get("event") == "shadow_pair"}),
        "fits": fits,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(profile, indent=2) + "\n")


if __name__ == "__main__":
    main()
