from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


class CodeExecutor:
    def __init__(self, scratch_dir: Path, timeout: float = 60) -> None:
        self.scratch_dir = scratch_dir
        self.timeout = timeout
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

    def run(self, code: str) -> tuple[str, str]:
        with tempfile.NamedTemporaryFile("w", dir=self.scratch_dir, suffix=".py", delete=False) as f:
            f.write(code)
            script_path = Path(f.name)
        try:
            result = subprocess.run(
                ["python3", str(script_path)],
                cwd=self.scratch_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired as e:
            return (e.stdout or ""), f"timed out after {self.timeout}s"
        finally:
            script_path.unlink(missing_ok=True)
