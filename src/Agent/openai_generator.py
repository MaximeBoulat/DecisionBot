from __future__ import annotations

import json
import urllib.request

from Agent.generator import ProposedStep
from Agent.tools import TOOLS_SCHEMA
from models import AgentStep, Task

SYSTEM_PROMPT = (
    "You are solving a data-analysis task about merchant payment processing fees. Use the provided "
    "tools to query transactions and compute fees - never guess a number without calling a tool to "
    "get it. When you have the final answer, reply with it directly as plain text, formatted exactly "
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

    def _input(self, task: Task, history: tuple[AgentStep, ...]) -> list[dict]:
        docs = "\n\n".join(f"# {name}\n{text}" for name, text in self.documentation.items())
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Documentation:\n{docs}\n\n"
            f"Question: {task.question}\n"
            f"Answer format: {task.answer_format}"
        )
        items: list[dict] = [{"role": "user", "content": prompt}]
        for s in history:
            if s.decided_by != "llm":
                continue
            call_id = f"call_{s.index}"
            items.append({"type": "function_call", "call_id": call_id, "name": s.tool_name,
                          "arguments": json.dumps(s.tool_args)})
            items.append({"type": "function_call_output", "call_id": call_id, "output": s.tool_result})
        return items

    def propose_step(self, task: Task, history: tuple[AgentStep, ...]) -> ProposedStep:
        payload = {"model": self.model, "input": self._input(task, history), "tools": TOOLS_SCHEMA}
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json", "authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.loads(r.read())
        for item in body.get("output", []):
            if item.get("type") == "function_call":
                return ProposedStep(action="call_tool", tool_name=item["name"],
                                     tool_args=json.loads(item["arguments"]))
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        return ProposedStep(action="finish", answer=content["text"].strip())
                    if content.get("type") == "refusal":
                        raise ValueError(f"model refused: {content.get('refusal')}")
        raise ValueError(f"no function_call or message in response: {json.dumps(body)}")
