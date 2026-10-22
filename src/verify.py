"""Step verification: ask checkers about each step of recorded episodes, then score them against step labels.

    uv run src/verify.py build  --episodes results/episodes.jsonl --run-dir out/verify/<name>
    uv run src/verify.py tokens --run-dir out/verify/<name> --gguf <model.gguf>
    uv run src/verify.py ask    --run-dir out/verify/<name> --checker decision --model <M> --url <U> [--dry-run]
    uv run src/verify.py report --run-dir out/verify/<name>

States hold only what the agent had plus facts computed in code; labels are written to a separate file that
`ask` never reads (it reads only whether a step has a label, to skip unlabeled ones). Odd task ids are for
development, even ones are held out.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from config import ANSWER_KEY_PATH, JEV_URL, OPENAI_API_KEY, OPENAI_URL, TYPESAFE_API_KEY
from Scorer.scorer import Scorer
from Scorer.step_labels import MERCHANTS, MONTHS, episode_points, label, seeds
from Verifier.checkers import DecisionModelChecker, OpenAIReviewerChecker, ReviewerChecker, VerdictCache
from Verifier.questions import flag_score, questions_for
from Verifier.state import STATE_VERSION, build_state

TOKENIZE = Path.home() / "AI/llama.cpp-pr29030-v070/build/bin/llama-tokenize"
FALSE_ALARM = 0.10      # thresholds flag at most this share of correct dev steps
MIN_WRONG = 5           # AUC lines with fewer wrong steps are not printed


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def cmd_build(a) -> None:
    scorer = Scorer(ANSWER_KEY_PATH)
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    points, labels = [], []
    for episode in read_jsonl(Path(a.episodes)):
        for point in episode_points(episode):
            verdict, reasons = label(point, scorer)
            items = [(point, "natural", None, verdict, reasons)]
            if verdict == "ok":
                for seeded, kind in seeds(point):
                    seeded_verdict, seeded_reasons = label(seeded, scorer)
                    if seeded_verdict != "wrong":
                        sys.exit(f"seed {seeded.point_id} is not labeled wrong ({seeded_verdict}); fix the seeding")
                    items.append((seeded, "seeded", kind, seeded_verdict, seeded_reasons))
            for p, provenance, kind, verdict_, reasons_ in items:
                state, meta = build_state(p, episode["documentation"])
                points.append({"point_id": p.point_id, "task_id": p.task.task_id, "tool": p.tool_name,
                               "provenance": provenance, "seed": kind, "state_version": STATE_VERSION,
                               "state": state, **meta})
                labels.append({"point_id": p.point_id, "label": verdict_, "reasons": reasons_})
    (run / "points.jsonl").write_text("".join(json.dumps(p) + "\n" for p in points))
    (run / "labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in labels))
    count = defaultdict(int)
    for p, x in zip(points, labels):
        count[(p["provenance"], "answer" if p["tool"] == "finish" else p["tool"], x["label"])] += 1
    for key in sorted(count, key=str):
        print(f"  {key[0]:8s} {key[1]:18s} {str(key[2]):6s} {count[key]}")
    print(f"{len(points)} points written to {run}")


def cmd_tokens(a) -> None:
    run = Path(a.run_dir)
    counts = {}
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
        for p in read_jsonl(run / "points.jsonl"):
            f.seek(0)
            f.truncate()
            f.write(json.dumps(p["state"]) + json.dumps(questions_for(p["tool"])))
            f.flush()
            out = subprocess.run([str(TOKENIZE), "-m", a.gguf, "-f", f.name, "--show-count", "--log-disable"],
                                 capture_output=True, text=True).stdout
            counts[p["point_id"]] = int(re.search(r"Total number of tokens: (\d+)", out).group(1))
    ordered = sorted(counts.values())
    print(f"{len(ordered)} states with questions: median {ordered[len(ordered) // 2]:,} tokens, max {ordered[-1]:,}; "
          f"over {a.context - 512:,}: {sum(v > a.context - 512 for v in ordered)} (context {a.context:,})")


def cmd_ask(a) -> None:
    run = Path(a.run_dir)
    labeled = {x["point_id"] for x in read_jsonl(run / "labels.jsonl") if x["label"] is not None}
    points = [p for p in read_jsonl(run / "points.jsonl") if (a.all or p["point_id"] in labeled)
              and (a.split == "all" or p["task_id"] % 2 == (1 if a.split == "dev" else 0))]
    if a.checker == "decision":
        url = a.url or JEV_URL
        checker = DecisionModelChecker(a.model, url, a.timeout, TYPESAFE_API_KEY if "typesafe.ai" in url else None)
    elif a.checker == "openai":
        if not OPENAI_API_KEY:
            sys.exit("OPENAI_API_KEY is not set (.env)")
        checker = OpenAIReviewerChecker(a.model, a.url or OPENAI_URL, OPENAI_API_KEY, a.timeout, a.effort)
    else:
        if not a.url:
            sys.exit("--url is required for a local reviewer")
        checker = ReviewerChecker(a.model, a.url, a.timeout)
    cache = VerdictCache(run / "verdicts.jsonl")
    todo = [p for p in points if cache.get(p["point_id"], checker.name, questions_for(p["tool"]), p["state"]) is None]
    cached = len(points) - len(todo)
    if a.limit:
        todo = todo[:a.limit]
    print(f"{checker.name}: {cached} cached, {len(todo)} to ask")
    if a.dry_run:
        for p in todo[:2]:
            print(json.dumps({k: v for k, v in p["state"].items() if k != "documentation"}, indent=1)[:3000])
        return
    if not todo:
        return
    print(f"canary ok: {checker.canary()}", flush=True)
    errors = 0
    for i, p in enumerate(todo, 1):
        row = cache.ask(checker, p["point_id"], p["state"], questions_for(p["tool"]))
        errors += "error" in row
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} ({errors} errors)", flush=True)


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def auc_ci(rows: list[tuple[int, float, bool]], reps: int = 1000) -> tuple:
    """AUC of score for wrong vs ok, with a 95% interval from resampling whole tasks."""
    point = auc([s for _, s, w in rows if w], [s for _, s, w in rows if not w])
    by_task = defaultdict(list)
    for row in rows:
        by_task[row[0]].append(row)
    tasks, rng, sims = list(by_task), random.Random(0), []
    for _ in range(reps):
        sample = [r for t in rng.choices(tasks, k=len(tasks)) for r in by_task[t]]
        value = auc([s for _, s, w in sample if w], [s for _, s, w in sample if not w])
        if value is not None:
            sims.append(value)
    sims.sort()
    ci = (sims[int(0.025 * len(sims))], sims[int(0.975 * len(sims)) - 1]) if len(sims) > 20 else (None, None)
    return point, ci


def given_up(p: dict) -> bool:
    return p["tool"] == "finish" and p["state"]["proposed_step"]["final_answer"].strip().lower() in ("", "not applicable")


def scope_fault(question: str, args: dict) -> bool:
    """Merchant, month, day, fee ID, rate or MCC named in the question but not matched by the arguments, read from
    words in the question rather than its template; or any exclusion filter."""
    named = [m for m in MERCHANTS if m in question]
    if (named and args.get("merchant") not in named) or (not named and args.get("merchant")):
        return True
    months = [MONTHS[w] for w in re.findall(r"[A-Z][a-z]+", question) if w in MONTHS]
    day = re.search(r"(\d+)(?:st|nd|rd|th) of the year", question)
    lo, month = args.get("day_of_year_min"), args.get("month")
    if day and (lo, args.get("day_of_year_max")) != (int(day[1]), int(day[1])):
        return True
    if months and ((month is not None and month != months[0]) or (month is None and lo is None)):
        return True
    if not months and not day and (month is not None or lo is not None):
        return True
    scenario = args.get("scenario") or {}
    for pattern, key in ((r"ID ?=? ?(\d+)", "fee_rule_id"), (r"changed to (\d+)", "rate"),
                         (r"MCC code to (\d+)", "merchant_category_code")):
        m = re.search(pattern, question)
        if m and key in scenario and scenario[key] != int(m[1]):
            return True
    return bool(args.get("exclude_refused") or args.get("card_scheme"))


def rule_scores(p: dict) -> dict[str, float]:
    """No-model baselines, computed from the state alone."""
    proposed = p["state"]["proposed_step"]
    if proposed["kind"] == "final_answer":
        return {"rule:not_applicable": float(given_up(p)),
                "rule:unsourced": float(any("does not appear" in f or "contains no number" in f
                                            for f in p["state"]["facts"]))}
    args = proposed["arguments"]
    sql = str(args.get("sql") or "").lower()
    rules = {"rule:exclusion": float(bool(args.get("exclude_refused") or args.get("exclude_fraudulent")
                                          or args.get("card_scheme") or re.search(r"refused|fraud", sql)))}
    if p["tool"] == "compute_fees":
        rules["rule:scope"] = float(scope_fault(p["state"]["question"], args))
    return rules


def checker_scores(path: Path, points: dict, labels: dict) -> dict:
    """(checker, question) -> point_id -> flag score for one verdicts file, plus each checker's mean and max."""
    scores = defaultdict(dict)
    for row in read_jsonl(path):
        p = points.get(row["point_id"])
        if "answers" not in row or p is None or row["point_id"] not in labels:
            continue
        flags = []
        for name, q in questions_for(p["tool"]).items():
            if name == "quantity" and given_up(p):
                continue    # 'not_observable' is the right reply to a give-up
            s = flag_score(q, row["answers"].get(name, {}), p["state"]["proposed_step"].get("final_answer"))
            if s is not None:
                scores[(row["checker"], name)][row["point_id"]] = s
                flags.append(s)
        if flags:
            scores[(row["checker"], "mean")][row["point_id"]] = statistics.mean(flags)
            scores[(row["checker"], "max")][row["point_id"]] = max(flags)
    return scores


def threshold(table: dict[str, float], ok_ids: list[str]) -> tuple[float, float]:
    """The lowest score that flags at most FALSE_ALARM of the given correct steps (a step is flagged when its score
    is at or above it), and the share it does flag. Ties at the top can make the target unreachable: then nothing
    is flagged."""
    ok = [table[i] for i in ok_ids if i in table]
    for t in sorted(set(table.values())) + [float("inf")]:
        fpr = sum(v >= t for v in ok) / len(ok)
        if fpr <= FALSE_ALARM:
            return t, fpr
    return float("inf"), 0.0


def cmd_report(a) -> None:
    run = Path(a.run_dir)
    points = {p["point_id"]: p for p in read_jsonl(run / "points.jsonl")}
    labels = {x["point_id"]: x["label"] for x in read_jsonl(run / "labels.jsonl") if x["label"] is not None}
    runs = [checker_scores(Path(f), points, labels) for f in (a.verdicts or [run / "verdicts.jsonl"])]
    scores = defaultdict(dict)      # (checker, question) -> point_id -> score, averaged over runs
    for key in {k for r in runs for k in r}:
        for pid in {pid for r in runs for pid in r.get(key, {})}:
            scores[key][pid] = statistics.mean(r[key][pid] for r in runs if pid in r.get(key, {}))
    for pid in labels:
        for name, s in rule_scores(points[pid]).items():
            scores[(name, "rule")][pid] = s
    wrong = lambda pid: labels[pid] == "wrong"
    dev = lambda pid: points[pid]["task_id"] % 2 == 1
    natural = lambda p: p["provenance"] == "natural"
    groups = {
        "natural calls": lambda p: natural(p) and p["tool"] != "finish",
        "natural answers, give-ups excluded": lambda p: natural(p) and p["tool"] == "finish" and not given_up(p),
        "natural give-ups (answers that are 'Not Applicable')": lambda p: natural(p) and given_up(p),
        "seeded calls": lambda p: p["tool"] != "finish",
        "seeded answers": lambda p: p["tool"] == "finish",
    }
    for kind in sorted({p["seed"] for p in points.values() if p["seed"]}):
        is_answer = kind in ("give_up", "quantity")
        groups[f"seeded {'answers' if is_answer else 'calls'}: {kind}"] = \
            lambda p, k=kind, a=is_answer: (p["tool"] == "finish") == a and p["seed"] in (None, k)
    if len(runs) > 1:
        print(f"Scores are averaged over {len(runs)} runs; 'runs' gives the lowest and highest single-run AUC.")
    report = {}
    for group, member in groups.items():
        seeded = group.startswith("seeded")
        print(f"\n== {group} (AUC wrong vs ok; {'seeded faults against correct originals' if seeded else 'natural steps'})")
        for split, parity in (("dev", 1), ("held-out", 0)):
            ids = [pid for pid, lab in labels.items() if points[pid]["task_id"] % 2 == parity and member(points[pid])
                   and (points[pid]["provenance"] == "seeded" or lab == "ok" if seeded else True)]
            n_wrong = sum(map(wrong, ids))
            print(f"  {split}: {n_wrong} wrong, {len(ids) - n_wrong} ok"
                  + ("" if n_wrong >= MIN_WRONG and n_wrong < len(ids) else "  (too few to report)"))
            if n_wrong < MIN_WRONG or n_wrong == len(ids):
                continue
            for key in sorted(scores, key=str):
                rows = [(points[i]["task_id"], scores[key][i], wrong(i)) for i in ids if i in scores[key]]
                if sum(w for *_, w in rows) < MIN_WRONG or all(w for *_, w in rows):
                    continue
                value, (lo, hi) = auc_ci(rows)
                report[f"{group}|{split}|{key[0]}|{key[1]}"] = {"auc": value, "ci": [lo, hi], "n": len(rows)}
                ci = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else "[n/a]"
                spread = ""
                if len(runs) > 1 and key[1] != "rule":
                    each = [auc([r[key][i] for i in ids if i in r.get(key, {}) and wrong(i)],
                                [r[key][i] for i in ids if i in r.get(key, {}) and not wrong(i)]) for r in runs]
                    each = [v for v in each if v is not None]
                    spread = f"  runs {min(each):.2f}-{max(each):.2f}" if each else ""
                print(f"    {key[0]:34s} {key[1]:11s} AUC {value:.3f} {ci}  n={len(rows)}{spread}")

    # Thresholds: each checker's score for calls and for answers is the question, mean or max with the best dev AUC,
    # and flags at most FALSE_ALARM of correct dev steps. Everything after is held-out.
    print(f"\n== Held-out catches at thresholds set on dev (each flags at most {FALSE_ALARM:.0%} of correct dev steps)")
    kinds = {"calls": lambda p: p["tool"] != "finish", "answers": lambda p: p["tool"] == "finish"}
    pools = {"natural, give-ups excluded": lambda p: natural(p) and not given_up(p),
             "natural give-ups": lambda p: natural(p) and given_up(p),
             "seeded": lambda p: not natural(p)}
    caught = defaultdict(dict)      # checker -> point_id -> flagged, for held-out wrong steps
    for c in sorted({k[0] for k in scores}):
        for kind, member in kinds.items():
            dev_ids = [i for i in labels if dev(i) and member(points[i])]
            keys = sorted(k for k in scores if k[0] == c and any(i in scores[k] for i in dev_ids))
            dev_auc = {k: auc([scores[k][i] for i in dev_ids if i in scores[k] and wrong(i)],
                              [scores[k][i] for i in dev_ids if i in scores[k] and not wrong(i)]) for k in keys}
            dev_auc = {k: v for k, v in dev_auc.items() if v is not None}
            if not dev_auc:
                continue
            key = max(dev_auc, key=dev_auc.get)
            t, fpr = threshold(scores[key], [i for i in dev_ids if not wrong(i)])
            held = [i for i in labels if not dev(i) and member(points[i]) and i in scores[key]]
            held_ok = [i for i in held if not wrong(i)]
            for i in held:
                if wrong(i):
                    caught[c][i] = scores[key][i] >= t
            counts = []
            for pool, in_pool in pools.items():
                ws = [i for i in held if wrong(i) and in_pool(points[i])]
                if ws:
                    counts.append(f"{pool} {sum(caught[c][i] for i in ws)}/{len(ws)}")
            held_fpr = sum(scores[key][i] >= t for i in held_ok) / len(held_ok) if held_ok else float("nan")
            print(f"  {c:34s} {kind:7s} {key[1]:11s} threshold {t:.3f} (dev false alarms {fpr:.0%}, held-out "
                  f"{held_fpr:.0%}); caught " + ", ".join(counts))

    print("\n== Held-out wrong steps both checkers miss (pairs of model checkers)")
    models = sorted(c for c in caught if not c.startswith("rule:"))
    for pool, in_pool in pools.items():
        for i, x in enumerate(models):
            for y in models[i + 1:]:
                ids = [p for p in caught[x] if p in caught[y] and in_pool(points[p])]
                if not ids:
                    continue
                mx, my = {p for p in ids if not caught[x][p]}, {p for p in ids if not caught[y][p]}
                note = "" if len(ids) >= 10 else "  (under 10 wrong steps: not interpreted)"
                print(f"  {pool}: {x} misses {len(mx)}, {y} misses {len(my)}, both {len(mx & my)} "
                      f"(independent: {len(mx) * len(my) / len(ids):.1f}) of {len(ids)}{note}")
    (run / "report.json").write_text(json.dumps(report, indent=1) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--episodes", required=True)
    b.add_argument("--run-dir", required=True)
    t = sub.add_parser("tokens")
    t.add_argument("--run-dir", required=True)
    t.add_argument("--gguf", required=True)
    t.add_argument("--context", type=int, default=16384)
    k = sub.add_parser("ask")
    k.add_argument("--run-dir", required=True)
    k.add_argument("--checker", choices=["decision", "openai", "reviewer"], required=True,
                   help="decision: a /v1/systemone model (Jev by default); openai: an OpenAI model through the "
                        "Responses API (Luna); reviewer: a local chat model scored by token log-probabilities")
    k.add_argument("--model", required=True)
    k.add_argument("--effort", help="openai only: the reasoning effort to pin, such as medium")
    k.add_argument("--url", help="defaults to Jev's or OpenAI's URL from config.py")
    k.add_argument("--timeout", type=float, default=300)
    k.add_argument("--limit", type=int)
    k.add_argument("--split", choices=["dev", "held-out", "all"], default="all")
    k.add_argument("--all", action="store_true", help="also ask about steps without an exact label")
    k.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--verdicts", nargs="+", help="verdict files of repeated runs on the same points; each step's "
                                                  "score is averaged across them (default: the run dir's own)")
    a = p.parse_args()
    {"build": cmd_build, "tokens": cmd_tokens, "ask": cmd_ask, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
