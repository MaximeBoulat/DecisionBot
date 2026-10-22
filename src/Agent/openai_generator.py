from __future__ import annotations

import json
import time

from Agent.arguments import matches, merge_arguments, missing_schema

from Agent.generator import ProposedStep
from Agent.retry import post_json
from Agent.tools import TOOLS_SCHEMA
from debug_log import DebugLog
from models import AgentStep, Task

SYSTEM_PROMPT = (
    "You are solving a data-analysis task about merchant payment processing fees. Use the provided "
    "tools to query transactions and compute fees - never guess a number without calling a tool to "
    "get it. query_data provides read-only SQL over all five raw files; call it with sql=null for schemas. "
    "Use it for whole-dataset statistics, joins, percentiles, correlation, rule-table queries and exact decimal "
    "rounding. Inspect pagination metadata; never treat a page of rows as the complete population. "
    "compute_fees applies the manual's fee rules and always reports a total; answer Not Applicable only "
    "when no tool can compute the answer. "
    "When you have the final answer, reply with it directly as plain text, formatted exactly "
    "per the answer format, and do not call a tool on that turn."
)


class OpenAIActionModel:
    def __init__(
        self,
        model: str,
        api_key: str,
        documentation: dict[str, str],
        base_url: str = "https://api.openai.com/v1/responses",
        timeout: float = 120,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.documentation = documentation
        self.base_url = base_url
        self.timeout = timeout

    def _task_prompt(self, task: Task) -> str:
        docs = "\n\n".join(f"# {name}\n{text}" for name, text in self.documentation.items())
        return (
            f"Documentation:\n{docs}\n\n"
            f"Question: {task.question}\n"
            f"Answer format: {task.answer_format}"
        )

    def _input(self, task: Task, history: tuple[AgentStep, ...]) -> list[dict]:
        items: list[dict] = [
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._task_prompt(task)},
        ]
        for s in history:
            call_id = f"call_{s.index}"
            items.append({"type": "function_call", "call_id": call_id, "name": s.tool_name,
                          "arguments": json.dumps(s.tool_args)})
            items.append({"type": "function_call_output", "call_id": call_id, "output": s.tool_result})
        return items

    def propose_step(self, task: Task, history: tuple[AgentStep, ...], debug_log: DebugLog) -> ProposedStep:
        if not history:
            debug_log.write("task_instructions", task_id=task.task_id, prompt=self._task_prompt(task))
        payload = {"model": self.model, "input": self._input(task, history), "tools": TOOLS_SCHEMA, "parallel_tool_calls": False}
        return self._request(payload, history, debug_log)

    def _request(self, payload: dict, history: tuple[AgentStep, ...], debug_log: DebugLog) -> ProposedStep:
        debug_log.write("llm_request", step=len(history), payload=payload)
        started = time.perf_counter()
        body = post_json(
            self.base_url, payload,
            {"content-type": "application/json", "authorization": f"Bearer {self.api_key}"},
            self.timeout, on_retry=lambda **retry: debug_log.write("http_retry", step=len(history), **retry),
        )
        debug_log.write("llm_response", step=len(history), body=body, elapsed_seconds=time.perf_counter() - started)
        usage = body.get("usage")
        for item in body.get("output", []):
            if item.get("type") == "function_call":
                return ProposedStep(action="call_tool", tool_name=item["name"],
                                     tool_args=json.loads(item["arguments"]), usage=usage)
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        return ProposedStep(action="finish", answer=content["text"], usage=usage)
                    if content.get("type") == "refusal":
                        raise ValueError(f"model refused: {content.get('refusal')}")
        raise ValueError(f"no function_call or message in response: {json.dumps(body)}")

    def complete_step(self, task: Task, history: tuple[AgentStep, ...], tool_name: str,
                      fixed: dict, debug_log: DebugLog) -> ProposedStep:
        items = self._input(task, history)
        if tool_name == "finish":
            items.append({"role": "user", "content": "Submit the final answer now, exactly in the required format."})
            return self._request({"model": self.model, "input": items, "tools": [], "tool_choice": "none"}, history, debug_log)
        tool = next(t for t in TOOLS_SCHEMA if t["name"] == tool_name)
        remaining = missing_schema(tool["parameters"], fixed)
        arguments = fixed
        if remaining["properties"]:
            items.append({"role": "user", "content": f"The selected tool is {tool_name}. Accepted arguments: {json.dumps(fixed)}. Supply only the remaining arguments defined by the tool schema."})
            partial_tool = {**tool, "parameters": remaining}
            proposed = self._request({
                "model": self.model, "input": items, "tools": [partial_tool],
                "tool_choice": {"type": "function", "name": tool_name}, "parallel_tool_calls": False,
            }, history, debug_log)
            if proposed.action != "call_tool" or proposed.tool_name != tool_name or proposed.tool_args is None:
                raise ValueError("Generator did not supply arguments for the selected tool")
            if not matches(proposed.tool_args, remaining):
                raise ValueError("Generated arguments do not match the remaining schema")
            arguments = merge_arguments(fixed, proposed.tool_args)
        if not matches(arguments, tool["parameters"]):
            raise ValueError("Completed arguments do not match the tool schema")
        return ProposedStep("call_tool", tool_name, arguments)
