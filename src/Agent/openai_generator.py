from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from Agent.generator import ProposedStep
from models import AgentStep, Task

SYSTEM_PROMPT = (
    "You are solving a data-analysis task by writing and running Python (pandas) code against the "
    "provided files. At each turn, either run one more piece of code, or finish with a final answer.\n\n"
    "Set `action` to `run_code` and `code` to the Python code to run, printing whatever you need to "
    "see, leaving `answer` null. Or set `action` to `finish` and `answer` to the final answer only, "
    "formatted exactly per the task's answer format, leaving `code` null."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["run_code", "finish"]},
        "code": {"type": ["string", "null"]},
        "answer": {"type": ["string", "null"]},
    },
    "required": ["action", "code", "answer"],
    "additionalProperties": False,
}


class OpenAICodeGenerationModel:
    def __init__(
        self,
        model: str,
        api_key: str,
        documentation: dict[str, str],
        data_file_paths: dict[str, Path],
        base_url: str = "https://api.openai.com/v1/responses",
        timeout: float = 120,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.documentation = documentation
        self.data_file_paths = data_file_paths
        self.base_url = base_url
        self.timeout = timeout

    def _prompt(self, task: Task, history: tuple[AgentStep, ...]) -> str:
        docs = "\n\n".join(f"# {name}\n{text}" for name, text in self.documentation.items())
        files = "\n".join(f"{name}: {path}" for name, path in self.data_file_paths.items())
        steps = "\n\n".join(
            f"[step {s.index}] code:\n{s.code}\nstdout:\n{s.stdout}\nstderr:\n{s.stderr}"
            + (f"\nproposed answer: {s.proposed_answer}\njev confidence: {s.jev_confidence}"
               if s.proposed_answer is not None else "")
            for s in history
        )
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Data files, by absolute path:\n{files}\n\n"
            f"Documentation:\n{docs}\n\n"
            f"Question: {task.question}\n"
            f"Answer format: {task.answer_format}\n\n"
            f"Steps so far:\n{steps if steps else '(none yet)'}"
        )

    def propose_step(self, task: Task, history: tuple[AgentStep, ...]) -> ProposedStep:
        payload = {
            "model": self.model,
            "input": self._prompt(task, history),
            "text": {"format": {"type": "json_schema", "name": "proposed_step", "schema": SCHEMA, "strict": True}},
        }
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json", "authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.loads(r.read())
        text = None
        for item in body.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text = content["text"]
                    elif content.get("type") == "refusal":
                        raise ValueError(f"model refused: {content.get('refusal')}")
        if text is None:
            raise ValueError(f"no message text in response: {json.dumps(body)}")
        parsed = json.loads(text)
        if parsed["action"] == "finish":
            return ProposedStep(action="finish", answer=parsed["answer"])
        return ProposedStep(action="run_code", code=parsed["code"])
