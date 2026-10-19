# DecisionBot

A live agent that solves [DABstep](https://huggingface.co/datasets/adyen/DABstep) data-analysis tasks by making tool calls against the real payments dataset, with the choice of which tool to call next, and the value of each of its bounded arguments, delegated to a decision model (Jev) whenever Jev is confident enough — the agent's own generation model (Sol) only decides what Jev defers. This reproduces the tool-and-argument cascade from "JevAgentHarness: Benchmarking, Deploying and Training Decision Models That Share Control with LLMs" (Zhu et al.), using the DABstep benchmark's tasks, data and documentation as the test environment, and Jev as the decision model.

DecisionBot's runs are scored against an independently computed answer key (`data/answer_key.jsonl`).

## Architecture

Two boundaries matter: the agent's own control plane, where a generation model and a decision model are both invoked but play different roles, and the Scorer, which is deliberately a separate system that never participates in solving a task — it only grades what the auv syncgent already recorded.

### Agent

```mermaid
flowchart TB
    Task["DABstep task"]
    subgraph Plane["DecisionBot — agent control plane"]
        subgraph Core["Core logic"]
            AgentLoop["Agent orchestration<br/>decide, delegate, dispatch, loop"]
            Context["Decision context<br/>task + step history"]
            Cascade["Cascade<br/>tool and argument delegation, by threshold"]
            subgraph Tools["Tools layer — contracts and callable capabilities"]
                QueryTool["query_transactions<br/>filter, group and aggregate payments.csv"]
                SqlTool["query_data<br/>read-only SQL over the raw DABstep tables"]
                FeesTool["compute_fees<br/>merchant-guide fee matching over fees.json"]
            end
        end
        subgraph Boundary["Model boundary"]
            GenAPI["Generation-model invocation"]
            DecisionAPI["Decision-model invocation"]
        end
    end
    OpenAI["OpenAI — GPT-6.1 Sol"]
    Jev["Jev — decision model"]
    Data[("DABstep data<br/>tasks, manual, payments")]
    Results[("Recorded episodes")]
    Task -->|"configures"| AgentLoop
    Data -->|"question, docs"| AgentLoop
    AgentLoop -->|"builds"| Context
    Context -->|"supplies state"| Cascade
    Cascade -->|"requests the tool and argument judgments"| DecisionAPI
    DecisionAPI -->|"asks"| Jev
    Jev -->|"returns a probability per option"| DecisionAPI
    DecisionAPI -->|"choice and confidence"| Cascade
    Cascade -->|"accepted choices bypass the generator"| AgentLoop
    Cascade -->|"deferred choices and free text"| GenAPI
    GenAPI -->|"calls"| OpenAI
    OpenAI -->|"returns a tool call, arguments or an answer"| GenAPI
    GenAPI -->|"returns proposal"| AgentLoop
    AgentLoop -->|"dispatches"| Tools
    QueryTool -->|"reads"| Data
    SqlTool -->|"reads"| Data
    FeesTool -->|"reads"| Data
    Tools -->|"result"| AgentLoop
    AgentLoop -->|"records"| Results
    classDef core fill:#e8f0fe,stroke:#4267a9
    classDef boundary fill:#edf0f4,stroke:#63758a
    classDef external fill:#FFE4B5,stroke:#FF8C00
    class AgentLoop,Context,Cascade,QueryTool,SqlTool,FeesTool core
    class GenAPI,DecisionAPI boundary
    class OpenAI,Jev,Data external
    style Results fill:#E3F2FD
```

The generation model (Sol) and the decision model (Jev) are invoked through the same kind of boundary but play different roles. At each step the cascade asks Jev which tool should be called next; if Jev's confidence on that choice clears the tool threshold, the choice is taken directly. For a chosen tool other than `finish`, the cascade separately asks Jev for the value of each bounded argument (one with a small, enumerable set of candidates drawn from the task and the step history), accepting each one whose confidence clears the argument threshold. Sol is only asked to decide what the cascade left open: the tool and arguments when Jev's tool confidence fell short, the remaining arguments when some were accepted and others were not, or the final answer text when the accepted choice is `finish`. The tools layer is three typed, fixed contracts: `query_transactions` and `compute_fees` cover filtering, aggregation and merchant-guide fee matching over the raw CSV/JSON files directly, and `query_data` runs one read-only SQL query over all of DABstep's tables for joins, percentiles and exact decimal arithmetic. Jev is never given a free-text channel — every judgment it returns is a probability over a small, enumerated set of options.

The tool and argument thresholds are not hand-set. They come from a calibration profile fitted by `src/calibrate.py` on shadow-mode runs: in shadow mode, the cascade still asks Jev for every tool and argument judgment and records its choice and confidence, but Sol always decides the actual step regardless of Jev's confidence. Whenever Jev's recorded choice disagrees with Sol's actual choice, DecisionBot forks the episode — it forces Jev's choice for that one decision and lets a baseline (Sol-only) continuation finish the task — and grades both the real episode and the forced continuation against the answer key, giving each decision a measured correctness label for Jev and for Sol. Paired with the measured dollar cost of the generator call that decision would have needed and an explicit, business-supplied dollar value for a correct task, `calibrate.py` sorts these labeled decisions into confidence bins and fits the lowest confidence above which every higher bin still shows a non-negative expected advantage to trusting Jev — the same rule the paper's cascade uses to set its thresholds.

```mermaid
flowchart TB
    Start@{ shape: start }
    BuildCtx["Build decision context<br/>from task and history"]
    AskTool["Cascade: ask Jev for the tool"]
    ToolGate{"Tool confidence ≥ threshold?"}
    AskArgs["Cascade: ask Jev each<br/>bounded argument's value"]
    NeedsGen{"Anything still<br/>left open?"}
    CompleteStep["Generator fills only<br/>what the cascade left open"]
    ProposeStep["Generator proposes<br/>the whole step"]
    Dispatch["Dispatch the tool call"]
    Observe["Record the step"]
    MaxSteps{"Max steps reached?"}
    Finish@{ shape: stop, label: "Episode finished<br/>answer recorded" }
    Fail@{ shape: stop, label: "Episode failed<br/>no answer" }
    Start --> BuildCtx --> AskTool --> ToolGate
    ToolGate -->|"no"| ProposeStep
    ToolGate -->|"yes"| AskArgs --> NeedsGen
    NeedsGen -->|"yes"| CompleteStep
    NeedsGen -->|"no"| Dispatch
    CompleteStep -->|"tool call"| Dispatch
    CompleteStep -->|"final answer"| Finish
    ProposeStep -->|"tool call"| Dispatch
    ProposeStep -->|"final answer"| Finish
    Dispatch --> Observe --> MaxSteps
    MaxSteps -->|"no"| BuildCtx
    MaxSteps -->|"yes"| Fail
    classDef core fill:#e8f0fe,stroke:#4267a9
    classDef gate fill:#fff2cc,stroke:#b58b24
    classDef terminal fill:#E3F2FD,stroke:#1565c0
    class BuildCtx,AskTool,AskArgs,CompleteStep,ProposeStep,Dispatch,Observe core
    class ToolGate,NeedsGen,MaxSteps gate
    class Finish,Fail terminal
```

Every step is recorded regardless of which side decided it, along with which Jev judgments were accepted and which were deferred, so the full episode — what Jev decided, what the generator filled in, and what finally got submitted — is reconstructible from `EpisodeResult` alone. `EpisodeResult` also carries the count of generator and Jev calls the episode actually made, so cascade runs can be compared against baseline (generator-only) runs on cost as well as success.

### Scoring

```mermaid
flowchart TB
    Results[("Recorded episodes")]
    AnswerKey[("Answer key")]
    subgraph ScorerPlane["Scorer"]
        Score["Grade episodes<br/>against the answer key"]
    end
    Results -->|"reads"| Score
    AnswerKey -->|"reads"| Score
    classDef scorer fill:#f3e5ff,stroke:#8326c4
    class Score scorer
    style Results fill:#E3F2FD
    style AnswerKey fill:#E3F2FD
```

The Scorer is a separate system from the agent: it never runs a task, it only reads the episodes the agent already recorded and the answer key, and grades one against the other.

## Usage

Run from the project folder in a terminal:

1. `cd` into this repository.
2. Create `.env` in the repository root (not committed — `.gitignore` excludes it) with:
   ```
   TYPESAFE_API_KEY=your-jev-key
   OPENAI_API_KEY=your-openai-key
   ```
3. `uv sync` — creates `.venv` and installs the project.
4. Run `src/main.py` with `--mode` set to one of:
   - `baseline` — the generator decides every step by itself; only `OPENAI_API_KEY` is required.
   - `shadow` — the cascade asks Jev for every tool and argument judgment and records its choice, but the generator still decides every step; requires `--task-value-usd`, the dollar value assigned to one correct task, which drives the correctness labels this mode produces for calibration.
   - `cascade` (default) — Jev's accepted judgments bypass the generator; requires either `--calibration <profile.json>` from a prior fit, or explicit `--tool-threshold`/`--argument-threshold` values.

   Each run creates its own timestamped folder under `out/` (or the path given with `--output-dir`), fetching the DABstep data on first use and writing `run.json`, `episodes.jsonl`, a per-task debug log under `debug/`, and `scores.json` once every task has run. `--task-ids` restricts a run to specific DABstep task IDs.
5. To fit a calibration profile from shadow-mode runs, point `src/calibrate.py` at their debug logs:
   ```
   uv run src/calibrate.py <shadow-run>/debug/task_*.jsonl --output <profile.json> \
     --generator-model gpt-6.1-sol --jev-model jev-latest
   ```
   The resulting profile can then be passed to a `cascade` run with `--calibration <profile.json>`.
6. To grade a previously recorded `episodes.jsonl` without making any model calls, run `src/main.py --score-only <path>`.

To run and debug any of this in VSCode instead: run `uv sync` once from the terminal first (VSCode doesn't do this for you), then open the Command Palette and run **Python: Select Interpreter**, choosing the `.venv` that `uv sync` just created. From there, open the Run and Debug panel and launch the **DecisionBot** configuration, or add the arguments above to its `args` list in `.vscode/launch.json` — breakpoints are available anywhere in the call chain.
