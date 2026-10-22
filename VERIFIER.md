# Step verification

This branch adds a second use of a decision model to DecisionBot. The shadow-mode cascade asks Jev to *choose* the agent's next action. Step verification asks a checker whether a step the agent has already proposed is *wrong*, before it takes effect. It then measures which wrong steps each checker catches and which ones different checkers miss together.

The first half of this file explains the test and how to run it. The second half is the analysis plan, registered on October 19, 2026, before any checker was asked about any step.

## Goal

Two questions, both about the steps of a data-analysis agent working on DABstep:

1. **Can a checker catch a wrong step before it takes effect?** A step is a proposed tool call (for example a `compute_fees` call with the wrong month) or a proposed final answer (for example "Not Applicable" when the tool results already hold the answer).
2. **Do a decision model and an LLM reviewer miss the same wrong steps?** If they fail together, adding the second checker buys little. This is the shared-blind-spot question in our research proposal, measured here step by step rather than only on final answers.

The intended panel is the two models DecisionBot already runs on: **Jev** (TypeSafe's decision model) and **Luna** (`gpt-6-luna`, OpenAI) as the LLM reviewer.

## Method

**Steps.** `build` replays every recorded main episode in `results/episodes.jsonl` (Max's 50-task shadow run) and turns each step into a verification point: the history the agent had at that moment, plus the step it proposed. The run has 399 steps.

**What a checker sees.** Only what the agent saw: the question, the answer format, the full documentation (`manual.md` and `payments-readme.md`, about 5,000 tokens), the earlier steps with their results, and the proposed step. It also sees a few facts computed in code:
- calendar dates for day and month arguments ("day of year 365 is December 31, 2023");
- for a final answer, which tool result each of its numbers came from ("3586.06 matches step 3 compute_fees result `baseline.total_fee`").

The last six steps show their results, each cut to about 1,500 characters (head and tail kept). Earlier steps show only the tool and its arguments. The same rule applies to every step. A checker never sees the answer key, the labels or any score.

**Questions.** Every question asks for one property, and each has an option that flags the step:

| Step | Question | Flags the step when |
|---|---|---|
| Tool call | `period`: does the call select the period the question asks about? | no |
| Tool call | `entity`: does it select the merchant the question asks about? | no |
| Tool call | `filter`: does it drop records that neither the question nor the documentation says to drop? | yes |
| Tool call | `change`: does a hypothetical change match only what the question says to change? | no |
| Tool call | `action`: proceed, revise, recheck or ask? | anything but proceed |
| Final answer | `quantity`: does it report the quantity asked for (a change, not a total)? | no |
| Final answer | `supported`: do the tool results support it? | no |
| Final answer | `answerable`: do the results already contain an answer? | yes, when the answer is "Not Applicable" |
| Final answer | `format`: does it follow the answer format? | no |
| Final answer | `action`: proceed, revise, recheck or ask? | anything but proceed |

Questions with a missing-evidence case have a "not applicable" or "not observable" option, so the checker is not forced into a yes or no.

**Labels (which steps are wrong).** Labels are made separately and never shown to a checker:
- **Final answers** are graded by the Scorer against the answer key.
- **`compute_fees` calls** on the 13 fee question templates are checked against the arguments the question implies: merchant, period, filters and any hypothetical change.

Every labeled step is one the agent actually took; the verifier plants no errors of its own. SQL queries (`query_data`) have no exact label and are skipped by default. In Max's first run that leaves 83 labeled steps: 45 fee calls and 38 answers.

**Checkers.** Each one answers the same questions about the same states, and each answer becomes a probability that the step is wrong:
- **Jev** gets one request per step with every question in it, and returns a probability per option.
- **Luna** gets one request per step through OpenAI's Responses API. A strict JSON schema asks it to *state* a probability for every option of every question. Those are stated probabilities, not probabilities read from the model's tokens. Luna returns reasoning, and OpenAI's reasoning models have not exposed token log-probabilities; this was not tested for Luna itself. Stated probabilities are coarser (many round numbers and ties) and less well calibrated, so a Jev-versus-Luna gap mixes the model with how its confidence is measured.

  *Option: token log-probabilities.* If the Responses API returns log-probabilities for Luna (request `"include": ["message.output_text.logprobs"]` with `"top_logprobs": 20`), Luna could be scored the way the local reviewer is: one single-letter reply per question, with option probabilities read from the first token's log-probabilities. That would put Jev and Luna on the same footing. It is not implemented. Try one request first, because a reasoning model may refuse the parameter or spend its first tokens on reasoning, and a reply per question costs about five times as many requests. If it is added, record which method produced each answer, so the two kinds of probability are never pooled.
- **No-model baselines** set the bar a checker must clear. `rule:exclusion` flags any call that excludes records, `rule:scope` flags a fee call whose merchant, month, day, fee ID, rate or MCC differs from the one the question names (read from words in the question, not its template) or that excludes records, `rule:not_applicable` flags an answer of "Not Applicable", and `rule:unsourced` flags an answer whose number appears in no tool result.

**Split.** Odd task ids are for development and even task ids are held out. Anything tuned, such as a question's wording or a threshold, is decided on development tasks, and claims come from held-out tasks.

## Running it

Put the agent's keys in `.env`, the same ones DecisionBot itself uses (see `.env.example`):

```sh
OPENAI_API_KEY=...
TYPESAFE_API_KEY=...
```

Then:

```sh
uv sync
R=out/verify/max-shadow-run

# 1. Turn the recorded run into steps, states and labels. No model calls.
uv run src/verify.py build --episodes results/episodes.jsonl --run-dir $R

# 2. Ask each checker about the labeled steps.
uv run src/verify.py ask --run-dir $R --checker decision --model jev-latest     # Jev, JEV_URL in config.py
uv run src/verify.py ask --run-dir $R --checker openai --model gpt-6-luna --effort medium   # Luna, OPENAI_URL in config.py

# 3. Score the checkers against the labels. No model calls.
uv run src/verify.py report --run-dir $R
```

What to expect:
- **`build`** prints counts by step type and label and writes `points.jsonl` (the states) and `labels.jsonl` (kept apart from the states).
- **`ask` sends a canary request first**: a question with a known answer. It stops before asking anything else if the canary comes back wrong, which catches a misconfigured endpoint or key.
- **Progress** prints every 10 steps.
- **Every answer is appended to `$R/verdicts.jsonl` as it arrives.** Rerunning the same command resumes where it stopped, and a finished run makes no new requests. The cache is keyed by the exact state and question set, so changing either one asks again rather than reusing stale answers.
- **`--split dev`, `--split held-out` and `--limit N`** run part of the set, for a quick first look. The full report needs both splits.
- **Pin Luna's reasoning effort** with `--effort`. The effort becomes part of the checker's name (`openai:gpt-6-luna@medium`), so answers at different efforts are never mixed, and the canary stops the run if the response reports a different effort.
- **Repeat a run in a new directory.** A rerun in the same directory replays the cache. Copy `points.jsonl` and `labels.jsonl` into a new directory, run `ask` there, and pass every run's file to the report: `report --run-dir $R --verdicts $R/verdicts.jsonl $R2/verdicts.jsonl`. Each step's score is averaged across runs, and each line also shows the lowest and highest single-run AUC. Luna's answers changed noticeably between two identical runs, so plan on three Luna runs.
- **Requests are sequential.** Plan on tens of minutes per checker.
- **Cost.** At the prices in `config.py` ($0.10 per million input tokens and $0.50 per million output tokens), a step is about 8,000 input tokens, so Luna's 83 steps on the first run are about $0.07 plus output and reasoning tokens. Jev's price is not recorded here.
- **On macOS, keep the machine awake** for the length of a run: `caffeinate -i -s -w <pid of the ask process>`.

**Local models (optional).** Any decision model behind a `/v1/systemone` endpoint works with `--checker decision --url <endpoint>`. Any OpenAI-compatible chat endpoint that returns `logprobs` works with `--checker reviewer --url <endpoint>`; it is scored from token log-probabilities, with thinking off. For example, with this project's llama.cpp router and llama-swap:

```sh
uv run src/verify.py ask --run-dir $R --checker decision --model vLLM-d3-Q4_K_M --url http://127.0.0.1:8092/v1/systemone
uv run src/verify.py ask --run-dir $R --checker reviewer --model Qwen3.8-27B --url http://127.0.0.1:9090/v1/chat/completions
```

Every state is under 11.1k tokens (d3 tokenizer), which fits Jev's 32k state limit and a 16k local context. `tokens --gguf <model.gguf>` counts them exactly; it expects llama.cpp's `llama-tokenize` at the path set in `src/verify.py`.

## Reading the report

`report` prints one block per group of steps, for development and held-out tasks separately:

- **calls**: the agent's tool calls, wrong or correct as recorded.
- **answers, give-ups excluded**: final answers other than "Not Applicable". This is where a checker has to judge an answer; give-ups are caught by `rule:not_applicable`.
- **give-ups**: final answers of "Not Applicable". In Max's first run all of them are wrong, so the group has no correct steps and no AUC.

Each line is one checker and question, or a checker's `mean` or `max` across its questions, followed by:
- **AUC:** the chance that a randomly chosen wrong step gets a higher flag than a randomly chosen correct one. 0.5 is chance, and 1.0 is perfect separation.
- **A 95% interval**, from resampling whole tasks.
- **n:** the number of steps behind it.
- **runs** (with several verdict files): the lowest and highest single-run AUC.

A split with fewer than 5 wrong steps is marked "too few to report" and has no lines. `quantity` is scored only on answers that are not "Not Applicable" (choosing `not_observable` for a give-up is correct), and `answerable` only on answers that are.

The `rule:` lines are the no-model baselines on the same steps. For a checker to be useful it must beat them, not just beat 0.5.

The next block gives **held-out catches**. For calls and for answers separately, each checker's score is the question, `mean` or `max` with the best development AUC. Its threshold is the lowest score that flags at most 10% of correct development steps; a step is flagged at or above it. The line shows the false-alarm rate the threshold reaches on development and held-out steps, and how many held-out wrong steps it catches: steps other than give-ups, and give-ups. A threshold of `inf` means ties at the top made 10% unreachable, so nothing is flagged.

The last lines are about **shared misses**, for each pair of model checkers and each of those two pools: how many held-out wrong steps each misses, how many both miss, and the number expected if their misses were independent. Both missing far more often than expected is a shared blind spot. Pools under 10 wrong steps are marked and not interpreted.

## Limits of this run

- **Max's run was recorded before the fee-tool fix**, so the states show the old `policy` argument and null totals.
- **The agent's own errors in this run are shallow.** All 4 wrong natural fee calls exclude refused payments, which `rule:exclusion` catches perfectly. Most of the 16 wrong natural answers are "Not Applicable", which `rule:not_applicable` mostly catches. Later runs on the dabstep-blindspots tasks have more errors of the agent's own, several identical to leaderboard agents' wrong answers.
- **This run checks the method and gives a first look.** Claims wait for fresh trajectories recorded with the fixed fee tool.

---

# Analysis plan (registered October 19, 2026)

Written before any checker was asked about any step. Changes after the first model run are listed at the end with their dates.

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
- **An LLM reviewer** through an OpenAI-compatible chat endpoint. It gives a single-token reply per question, with thinking off, and its option probabilities are read from token log-probabilities, so it is scored like a decision model.
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

## Changes after the first model run

- October 19, 2026, while the first local run (d3, 60 development steps) was in progress and before any result was read: added the `openai` checker so the panel can be Jev and Luna, the models DecisionBot runs on. Luna's scores are stated probabilities, not token probabilities (see Method).
- October 21, 2026, after reading two full Jev and Luna runs. These change the report, not the questions or states, so the cached answers stay valid:
  - **Shared misses.** A step whose score tied the threshold counted as missed, so Luna, whose stated probabilities reach 1.0 on most correct steps, got a threshold of 1.000 and "missed" every wrong step. Steps are now flagged at or above the threshold, and the report prints the false-alarm rate the threshold reaches. The pool also included seeded steps, though the plan names natural ones; natural steps, natural give-ups and seeded steps are now separate pools.
  - **Overall score.** The maximum over stated probabilities saturates, which made `any` Luna's weakest score. Each checker's overall score is now chosen on development steps from its questions, their mean and their maximum (`max` replaces `any`), separately for calls and answers.
  - **`quantity` and `answerable`.** `quantity` offered `not_observable` for give-ups, which is the right reply, but it was scored as not flagging them; it is now scored only on answers other than "Not Applicable". `answerable` was scored 0 on every other answer, which made it a copy of `rule:not_applicable`; it is now scored only on give-ups.
  - **Give-ups.** Every give-up in Max's run is wrong, so `rule:not_applicable` separates them perfectly and they inflated every answer-level line. Natural answers are now reported with give-ups excluded, and give-ups on their own.
  - **Small groups.** Lines with fewer than 5 wrong steps are no longer printed.
  - **`rule:scope`.** A no-model baseline that compares the merchant, month, day, fee ID, rate and MCC named in the question with the call's arguments. It scores 1.00 on every seeded call fault, so those seeds measure whether a question is sensitive, not whether a checker adds to code.
  - **Repeated runs.** Luna's AUCs moved by a median of 0.04 between two identical runs (Jev's by 0.004), which the task-resampled intervals do not show. `report` now averages several verdict files and prints the single-run range. Luna's reasoning effort is pinned with `--effort` and is part of the checker name.
- October 22, 2026: **seeded faults removed.** `build` no longer copies correct steps with a planted fault, and `report` scores only steps the agent took (in an older run directory, seeded points are skipped). The seeds measured whether a question reacts to an obvious fault, and `rule:scope` showed code catches all of them, so they could not show a checker adding judgment, and their lines in the report drew attention from the agent's own errors. What the verifier is for is whether checkers miss the same errors agents make, and that needs the agent's own errors: the October 20 and 21 runs on the dabstep-blindspots tasks have them, several identical to leaderboard agents' wrong answers. The registered plan above still describes seeds; this entry supersedes those parts.
