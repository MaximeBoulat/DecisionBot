from __future__ import annotations

import json
from pathlib import Path


class DebugLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def write(self, event: str, **fields) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps({"event": event, **fields}) + "\n")
