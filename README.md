# DecisionBot

A live agent that solves [DABstep](https://huggingface.co/datasets/adyen/DABstep) data-analysis tasks by making tool calls against the real payments dataset, with one decision point — whether a proposed answer is actually ready to submit — delegated to a decision model (Jev) instead of being decided by the agent's own LLM. This reproduces the delegation architecture from "JevAgentHarness: Benchmarking, Deploying and Training Decision Models That Share Control with LLMs" (Zhu et al.), using the DABstep benchmark's tasks, data and documentation as the test environment, and Jev as the decision model.

DecisionBot's runs are scored against an independently computed answer key (`data/answer_key.jsonl`).

## Architecture

Two boundaries matter: the agent's own control plane, where a generation model and a decision model are both invoked but play different roles, and the Scorer, which is deliberately a separate system that never participates in solving a task — it only grades what the agent already recorded.

### Agent

```mermaid
flowchart TB
    Start@{ shape: start }
    Task["DABstep task"]
    subgraph Plane["DecisionBot — agent control plane"]
        subgraph Core["Core logic"]
            AgentLoop["Agent orchestration<br/>propose, delegate, execute, loop"]
            Context["Decision context<br/>task + step history + proposed answer"]
            StopQ["Stop-guard question"]
            Executor["Code executor<br/>sandboxed subprocess"]
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
    Start --> AgentLoop
    Task -->|"configures"| AgentLoop
    Data -->|"question, docs, file paths"| AgentLoop
    AgentLoop -->|"requests the next step"| GenAPI
    GenAPI -->|"calls"| OpenAI
    OpenAI -->|"returns code or a proposed answer"| GenAPI
    GenAPI -->|"returns proposal"| AgentLoop
    AgentLoop -->|"runs proposed code"| Executor
    Executor -->|"executes against"| Data
    Executor -->|"returns stdout/stderr"| AgentLoop
    AgentLoop -->|"on a proposed finish, builds"| Context
    Context -->|"supplies evidence"| DecisionAPI
    StopQ -->|"defines the choice"| DecisionAPI
    DecisionAPI -->|"requests a judgment"| Jev
    Jev -->|"returns a probability"| DecisionAPI
    DecisionAPI -->|"accept or reject the stop"| AgentLoop
    AgentLoop -->|"records"| Results
    classDef core fill:#e8f0fe,stroke:#4267a9
    classDef boundary fill:#edf0f4,stroke:#63758a
    classDef external fill:#FFE4B5,stroke:#FF8C00
    class AgentLoop,Context,StopQ,Executor core
    class GenAPI,DecisionAPI boundary
    class OpenAI,Jev,Data external
    style Results fill:#E3F2FD
```

The generation model (Sol) and the decision model (Jev) are invoked through the same kind of boundary but play different roles: Sol does the actual work — it is the only thing that writes code or proposes an answer, since Jev can only answer a typed question with a probability, never generate free text. Jev is asked exactly one question, only when Sol proposes to stop: given the task, the full step history and the proposed answer, is this actually finished and correct? The agent only accepts the stop when Jev's confidence clears the threshold; otherwise the episode continues.

```mermaid
flowchart TB
    Start@{ shape: start }
    Step["Generator proposes a step"]
    Decide{"Proposed action"}
    RunCode["Execute code"]
    Observe["Record stdout, stderr<br/>as a step, decided_by: llm"]
    BuildCtx["Build decision context<br/>from task, history, proposed answer"]
    AskJev["Ask Jev the stop-guard question"]
    JevDecide{"Jev confidence ≥ threshold?"}
    RecordReject["Record rejected stop<br/>decided_by: jev"]
    Finish@{ shape: stop, label: "Episode finished<br/>answer recorded" }
    MaxSteps{"Max steps reached?"}
    Fail@{ shape: stop, label: "Episode failed<br/>no answer" }
    Start --> Step
    Step --> Decide
    Decide -->|"run_code"| RunCode
    RunCode --> Observe
    Observe --> MaxSteps
    Decide -->|"finish"| BuildCtx
    BuildCtx --> AskJev 
    AskJev --> JevDecide
    JevDecide -->|"yes"| Finish
    JevDecide -->|"no"| RecordReject
    RecordReject --> MaxSteps
    MaxSteps -->|"no"| Step
    MaxSteps -->|"yes"| Fail
    classDef core fill:#e8f0fe,stroke:#4267a9
    classDef gate fill:#fff2cc,stroke:#b58b24
    classDef terminal fill:#E3F2FD,stroke:#1565c0
    class Step,RunCode,Observe,BuildCtx,AskJev,RecordReject core
    class Decide,JevDecide,MaxSteps gate
    class Finish,Fail terminal
```

Every step is recorded regardless of outcome, including a rejected stop attempt, so the full episode — what was tried, what Jev overruled, and what finally got submitted — is reconstructible from `EpisodeResult` alone.

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
4. `uv run src/main.py` — fetches the DABstep data on first run, then runs the agent on one task end to end, printing the resulting `EpisodeResult` and appending it to `out/episodes.jsonl`.

To run and debug it in VSCode instead: run `uv sync` once from the terminal first (VSCode doesn't do this for you), then open the Command Palette and run **Python: Select Interpreter**, choosing the `.venv` that `uv sync` just created. From there, open the Run and Debug panel and launch the **DecisionBot** configuration — it runs `src/main.py` the same way, with breakpoints available anywhere in the call chain.

Scoring against the answer key (`data/answer_key.jsonl`) isn't wired into `main.py` yet — `Scorer.score()` exists and can be called once that file exists, but nothing currently tells you whether a run's answer was actually correct.
