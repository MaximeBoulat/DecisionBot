from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

from models import Task


class DabstepData:
    REVISION = "523e6627415c6ebafa0954ce0df9cd93b6c0914d"
    HF = f"https://huggingface.co/datasets/adyen/DABstep/resolve/{REVISION}/"
    DOCS = ["manual.md", "payments-readme.md"]
    DATA_FILES = ("payments.csv", "fees.json", "merchant_data.json", "merchant_category_codes.csv",
                  "acquirer_countries.csv")

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def _fetch_one(self, rel: str, dest: Path) -> None:
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(self.HF + urllib.parse.quote(rel)) as r:
            dest.write_bytes(r.read())

    def fetch(self) -> None:
        for name in self.DOCS + list(self.DATA_FILES):
            self._fetch_one(f"data/context/{name}", self.data_dir / "context" / name)
        self._fetch_one("data/tasks/all.jsonl", self.data_dir / "tasks.jsonl")

    def load_tasks(self) -> dict[int, Task]:
        tasks = {}
        for line in (self.data_dir / "tasks.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = int(row["task_id"])
            tasks[task_id] = Task(task_id=task_id, question=row["question"], answer_format=row["guidelines"])
        return tasks

    def load_documentation(self) -> dict[str, str]:
        return {name: (self.data_dir / "context" / name).read_text() for name in self.DOCS}

    def data_file_paths(self) -> dict[str, Path]:
        return {name: self.data_dir / "context" / name for name in self.DATA_FILES}
