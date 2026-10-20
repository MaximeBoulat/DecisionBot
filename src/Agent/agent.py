from __future__ import annotations

import json
from dataclasses import asdict
from typing import Literal

from Agent.arguments import argument_choices, get_value, missing_schema, put_value
from Agent.cascade import Cascade
from Agent.context import DecisionContext
from Agent.cost import generator_call_cost
from Agent.generator import ActionModel
from Agent.tool_schemas import TOOLS_SCHEMA
from Agent.tools import ToolsLayer
from debug_log import DebugLog
from models import AgentStep, DecisionQuestion, EpisodeResult, Task
from Scorer.scorer import Scorer


class Agent:
    def __init__(self, generator: ActionModel, cascade: Cascade, tools: ToolsLayer,
                 documentation: dict[str, str], mode: Literal["baseline", "shadow", "cascade"] = "cascade",
                 max_steps: int = 15, scorer: Scorer | None = None, task_value_usd: float | None = None,
                 total_tasks: int = 1) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if mode == "shadow" and (scorer is None or task_value_usd is None):
            raise ValueError("Shadow mode requires a scorer and a task_value_usd to label decisions")
        if task_value_usd is not None and task_value_usd <= 0:
            raise ValueError("task_value_usd must be positive")
        if total_tasks < 1:
            raise ValueError("total_tasks must be positive")
        self.generator = generator
        self.cascade = cascade
        self.tools = tools
        self.documentation = documentation
        self.mode = mode
        self.max_steps = max_steps
        self.scorer = scorer
        self.task_value_usd = task_value_usd
        self.total_tasks = total_tasks
        self.schemas = {tool["name"]: tool for tool in TOOLS_SCHEMA}
        self.tool_question = DecisionQuestion(
            "q0", "choice", "Which tool should the agent call next?",
            {**{name: tool["description"].splitlines()[0] for name, tool in self.schemas.items()},
             "finish": "Call when the task is complete. Provide the final answer (or a short summary)."},
        )

    def run(self, task: Task, debug_log: DebugLog, task_number: int = 1) -> EpisodeResult:
        history: list[AgentStep] = []
        shadow_pairs: list[dict] | None = [] if self.mode == "shadow" else None
        final_answer, error, llm_calls, jev_calls = self._execute(
            task, history, debug_log, 0, 0, shadow_pairs, task_number)
        if shadow_pairs is not None:
            llm_correct = final_answer is not None and error is None and self._grade(task, final_answer)
            for pair in shadow_pairs:
                pair["llm_correct"] = llm_correct
                if pair["jev_correct"] is None and pair["decision"]["choice"] != "other":
                    pair["jev_correct"] = llm_correct
                debug_log.write("shadow_pair", **pair)
        return EpisodeResult(
            task.task_id, tuple(history), final_answer, final_answer is not None, error,
            task=task, documentation=self.documentation, mode=self.mode,
            llm_calls=llm_calls, jev_calls=jev_calls,
        )

    def _grade(self, task: Task, answer: str) -> bool:
        detail = self.scorer.score([EpisodeResult(task.task_id, (), answer, True, None)])["tasks"][0]
        if detail["status"] == "ambiguous":
            return bool(detail["matched_interpretations"])
        return detail["status"] == "correct"

    def _fork_tool(self, task: Task, history: list[AgentStep], jev_tool: str, debug_log: DebugLog,
                   task_number: int) -> bool:
        if jev_tool == "finish":
            proposed = self.generator.complete_step(task, tuple(history), "finish", {}, debug_log)
            return proposed.answer is not None and self._grade(task, proposed.answer)
        proposed = self.generator.complete_step(task, tuple(history), jev_tool, {}, debug_log)
        output = self.tools.dispatch(jev_tool, proposed.tool_args)
        payload = json.loads(output)
        if isinstance(payload, dict) and payload.get("error"):
            return False
        step = AgentStep(len(history), jev_tool, proposed.tool_args, output, "jev")
        fork_agent = Agent(self.generator, self.cascade, self.tools, self.documentation,
                            "baseline", self.max_steps, self.scorer, self.task_value_usd, self.total_tasks)
        final_answer, error, _, _ = fork_agent._execute(task, history + [step], debug_log, 0, 0, None, task_number)
        return final_answer is not None and error is None and self._grade(task, final_answer)

    def _fork_argument(self, task: Task, history: list[AgentStep], tool_name: str, real_arguments: dict,
                        path: tuple[str, ...], jev_value: object, debug_log: DebugLog,
                        task_number: int) -> bool:
        forced = json.loads(json.dumps(real_arguments))
        put_value(forced, path, jev_value)
        output = self.tools.dispatch(tool_name, forced)
        payload = json.loads(output)
        if isinstance(payload, dict) and payload.get("error"):
            return False
        step = AgentStep(len(history), tool_name, forced, output, "jev")
        fork_agent = Agent(self.generator, self.cascade, self.tools, self.documentation,
                            "baseline", self.max_steps, self.scorer, self.task_value_usd, self.total_tasks)
        final_answer, error, _, _ = fork_agent._execute(task, history + [step], debug_log, 0, 0, None, task_number)
        return final_answer is not None and error is None and self._grade(task, final_answer)

    def _execute(self, task: Task, history: list[AgentStep], debug_log: DebugLog,
                 llm_calls: int, jev_calls: int,
                 shadow_pairs: list[dict] | None, task_number: int) -> tuple[str | None, str | None, int, int]:
        final_answer: str | None = None
        error: str | None = None
        for index in range(len(history), self.max_steps):
            decisions = []
            sources = {}
            tool_name = "generator"
            arguments: dict = {}
            origin = "llm"
            events = []
            line_started = False
            try:
                context = DecisionContext(task, self.documentation, tuple(history))
                tool_decision = None
                if self.mode != "baseline":
                    tool_decision = self.cascade.decide(context, "tool", self.tool_question, debug_log,
                                                         shadow=self.mode == "shadow")
                    decisions.append(tool_decision)
                    jev_calls += 1
                if tool_decision is not None and tool_decision.accepted:
                    tool_name = tool_decision.choice
                    origin = "jev"
                    if tool_name != "finish":
                        for candidate in argument_choices(self.schemas[tool_name]["parameters"], context):
                            name = ".".join(candidate.path)
                            question = DecisionQuestion(
                                "q0", "choice",
                                f"Which value should the argument `{name}` of tool `{tool_name}` take? "
                                f"({candidate.schema.get('description', '')})",
                                candidate.criteria,
                            )
                            record = self.cascade.decide(context, "argument", question, debug_log)
                            decisions.append(record)
                            jev_calls += 1
                            if record.accepted:
                                put_value(arguments, candidate.path, candidate.values[record.choice])
                                sources[name] = "jev"
                    needs_generator = tool_name == "finish" or bool(missing_schema(
                        self.schemas[tool_name]["parameters"], arguments)["properties"])
                    if needs_generator:
                        events.append("Asking generator for " + ("answer..." if tool_name == "finish" else "deferred arguments..."))
                        llm_calls += 1
                    print(f"Task {task.task_id} ({task_number}/{self.total_tasks}) — step {index + 1}/{self.max_steps}: " + " ".join(events), end="", flush=True)
                    line_started = True
                    events.clear()
                    proposed = self.generator.complete_step(task, tuple(history), tool_name, arguments, debug_log)
                else:
                    events.append("Asking generator...")
                    llm_calls += 1
                    print(f"Task {task.task_id} ({task_number}/{self.total_tasks}) — step {index + 1}/{self.max_steps}: " + " ".join(events), end="", flush=True)
                    line_started = True
                    events.clear()
                    proposed = self.generator.propose_step(task, tuple(history), debug_log)
                tool_name = "finish" if proposed.action == "finish" else proposed.tool_name
                arguments = proposed.tool_args or {}
                if self.mode == "shadow" and shadow_pairs is not None:
                    print(flush=True)
                    tool_saved_cost = generator_call_cost(proposed.usage)
                    tool_agrees = tool_decision.choice == tool_name
                    tool_jev_correct = None if tool_agrees or tool_decision.error else self._fork_tool(
                        task, history, tool_decision.choice, debug_log, task_number)
                    shadow_pairs.append({
                        "task_id": task.task_id, "step": index, "point": "tool",
                        "decision": asdict(tool_decision), "llm_choice": tool_name,
                        "jev_correct": tool_jev_correct, "llm_correct": None,
                        "saved_cost": tool_saved_cost, "error_value": self.task_value_usd,
                    })
                    if tool_name in self.schemas:
                        for candidate in argument_choices(self.schemas[tool_name]["parameters"], context):
                            name = ".".join(candidate.path)
                            question = DecisionQuestion(
                                "q0", "choice",
                                f"Which value should the argument `{name}` of tool `{tool_name}` take? "
                                f"({candidate.schema.get('description', '')})", candidate.criteria,
                            )
                            record = self.cascade.decide(context, "argument", question, debug_log, shadow=True)
                            decisions.append(record)
                            jev_calls += 1
                            llm_value = get_value(arguments, candidate.path)
                            if record.choice == "other" or record.error:
                                arg_jev_correct = None
                            else:
                                arg_agrees = candidate.values.get(record.choice) == llm_value
                                arg_jev_correct = None if arg_agrees else self._fork_argument(
                                    task, history, tool_name, arguments, candidate.path,
                                    candidate.values[record.choice], debug_log, task_number)
                            shadow_pairs.append({
                                "task_id": task.task_id, "step": index, "point": "argument", "argument": name,
                                "decision": asdict(record), "option_values": candidate.values,
                                "llm_choice": llm_value, "jev_correct": arg_jev_correct, "llm_correct": None,
                                "saved_cost": tool_saved_cost, "error_value": self.task_value_usd,
                            })
                if proposed.action == "finish":
                    if proposed.answer is None:
                        raise ValueError("Finish must contain an answer")
                    final_answer = proposed.answer
                    output = proposed.answer
                    events.append("Task finished.")
                    tool_error = None
                else:
                    if tool_name not in self.schemas or proposed.tool_args is None:
                        raise ValueError("Unknown tool or missing arguments")
                    print(f" Calling tool: {tool_name}", end="", flush=True)
                    output = self.tools.dispatch(tool_name, arguments)
                    payload = json.loads(output)
                    tool_error = payload.get("error") if isinstance(payload, dict) else None
                    if tool_error:
                        events.append("Failed: " + " ".join(str(tool_error).splitlines()))
                    debug_log.write("tool_call", step=index, name=tool_name, args=arguments, result=output)
                history.append(AgentStep(
                    index, tool_name, arguments, output, origin,
                    proposed_answer=final_answer, error=tool_error,
                    decisions=tuple(decisions), argument_sources=sources,
                ))
                print(" " + " ".join(events) if events else "", flush=True)
                if final_answer is not None:
                    break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                history.append(AgentStep(index, tool_name, arguments, json.dumps({"error": error}), origin,
                                         error=error, decisions=tuple(decisions), argument_sources=sources))
                debug_log.write("episode_error", step=index, error=error)
                prefix = " Failed — " if line_started else f"Task {task.task_id} ({task_number}/{self.total_tasks}) — step {index + 1}/{self.max_steps}: Failed — "
                print(prefix + " ".join(error.splitlines()), flush=True)
                break
        if final_answer is None and error is None:
            error = "max steps exceeded"
            print(f"Task {task.task_id} ({task_number}/{self.total_tasks}): Step limit reached.", flush=True)
        return final_answer, error, llm_calls, jev_calls
