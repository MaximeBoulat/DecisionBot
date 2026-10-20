# Step verification: analysis plan

Written on October 19, 2026, before any checker was asked about any step. Changes after the first model run are listed at the end with their dates.

## Question

Can a checker catch a wrong step of a data-analysis agent before it takes effect, and do decision models and LLM reviewers miss the same wrong steps? A step is a proposed tool call or a proposed final answer. The checker sees what the agent saw, plus facts computed in code. It never sees the answer key, step labels or scores.

## Units and labels

`src/verify.py build` turns each recorded main episode (`episodes.jsonl`) into one verification point per step. Labels are written to a separate file, and `ask` reads them only to skip steps that have no label.

| Label source | Steps | Wrong means |
|---|---|---|
| Final answers | every `finish` step | the Scorer grades the answer incorrect; ambiguous tasks have no label |
| Fee calls | `compute_fees` calls on the 13 fee templates with a fixed question shape (`Scorer/step_labels.py`) | merchant, period, a filter (`exclude_refused`, a card-scheme filter) or the hypothetical change departs from what the question implies; a call without a hypothetical change is not wrong for lacking one |
| Seeded faults | copies of correctly labeled steps with one fault: `period`, `merchant`, `filter`, `change` (calls); `give_up`, `quantity` (answers) | wrong by construction; `build` stops if a seed is not labeled wrong |

SQL calls (`query_data`) have no label and are not asked about by default.

## Checkers

- **Decision models** through a `/v1/systemone` endpoint, with all questions for a step in one request.
- **An LLM reviewer** through an OpenAI-compatible chat endpoint. It answers one single-token reply per question, with thinking off, and its option probabilities are read from token log-probabilities, so it is scored like a decision model.
- **No-model baselines** computed from the state: `rule:exclusion` (any exclusion filter on a call), `rule:not_applicable` (the answer is "Not Applicable" or empty), and `rule:unsourced` (a number in the answer appears in no tool result).

Every checker run starts with a canary request with a known answer and stops if it fails. Responses are cached by point, checker, question-set hash and exact state hash. State version 1. Question sets `09fddf2da9a0` (calls) and `4813fcc128c2` (answers).

## Measures

- **Score per question:** the probability of the option that flags the step (`Verifier/questions.py`). A checker's overall score for a step is the maximum over its questions (`any`).
- **AUC** of the score for wrong against correct steps, with a 95% interval from 1,000 resamples of whole tasks. It is reported for natural calls, natural answers, all seeded calls and all seeded answers, and for each fault type, with seeded faults compared against the correct originals.
- **Split:** odd task ids are for development, and even task ids are held out. Questions, thresholds and state changes are decided on development tasks only.
- **Shared misses:** each checker's threshold flags 10% of correct development steps. On held-out wrong steps, the report counts the steps both checkers of a pair miss, against the count expected if they missed independently.

## What counts as a result

- **A checker has a signal** on a group if the lower bound of its held-out AUC is above 0.5.
- **A checker adds to code** on a group only if its held-out AUC is above the best no-model baseline's on the same steps. Without a paired test, this is a description, not a claim.
- **A shared blind spot** is a held-out co-miss count above the independence expectation. Counts under 10 wrong steps are reported but not interpreted.

## Known limits of the first run (Max's 50-task shadow run)

- It was recorded before the fee-tool fix. States show the old `policy` argument and null totals.
- **Natural positives are few:** 4 wrong fee calls, all `exclude_refused`, which `rule:exclusion` catches perfectly, and 16 wrong final answers, mostly "Not Applicable", which `rule:not_applicable` mostly catches. Natural-step results from this run say little about judgment.
- **Seeded faults are simple and template-bound.** There are only 3 `quantity` seeds, too few to report.
- A brief local test on this run checks the pipeline and gives a first look. Claims wait for fresh trajectories on the fixed tool.

## Running the test

The run data is `results/episodes.jsonl` (Max's 50-task shadow run, October 2026). The checks need a decision model behind a `/v1/systemone` endpoint and, for the reviewer, an OpenAI-compatible chat endpoint that returns `logprobs`. Neither one is started by the scripts.

```sh
uv sync
R=out/verify/max-shadow-run
uv run src/verify.py build --episodes results/episodes.jsonl --run-dir $R    # 399 points, 233 with labels

# One checker at a time. --split dev|held-out|all, --limit N; reruns skip cached answers.
uv run src/verify.py ask --run-dir $R --checker decision --model vLLM-d3-Q4_K_M --url http://127.0.0.1:8092/v1/systemone
uv run src/verify.py ask --run-dir $R --checker decision --model jev-latest --url https://api.typesafe.ai/v1/systemone   # TYPESAFE_API_KEY in .env
uv run src/verify.py ask --run-dir $R --checker reviewer --model Qwen3.8-27B --url http://127.0.0.1:9090/v1/chat/completions

uv run src/verify.py report --run-dir $R    # no model calls
```

On macOS, keep the machine awake for the length of a run: `caffeinate -i -s -w <pid of the ask process>`. Progress is printed every 10 steps, and each answer is appended to `$R/verdicts.jsonl` as it arrives. `tokens --gguf <model.gguf>` counts state sizes exactly; it expects llama.cpp's `llama-tokenize` at the path set in `src/verify.py`.

## Changes after the first model run

None yet.
