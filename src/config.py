import os
from pathlib import Path


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env(Path(".env"))

DATA_DIR = Path("data").resolve()
OUT_PATH = Path("out") / "episodes.jsonl"
ANSWER_KEY_PATH = Path("answer_key.jsonl").resolve()
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
TYPESAFE_API_KEY = os.environ["TYPESAFE_API_KEY"]
OPENAI_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = "gpt-6.1-sol"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
