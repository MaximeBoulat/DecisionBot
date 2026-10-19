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


PROJECT_DIR = Path(__file__).resolve().parent.parent
_load_env(PROJECT_DIR / ".env")

DATA_DIR = PROJECT_DIR / "data"
OUT_PATH = PROJECT_DIR / "out" / "episodes.jsonl"
SCORE_PATH = PROJECT_DIR / "out" / "scores.json"
DEBUG_DIR = PROJECT_DIR / "out" / "debug"
ANSWER_KEY_PATH = PROJECT_DIR / "answer_key.jsonl"
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
TYPESAFE_API_KEY = os.environ.get("TYPESAFE_API_KEY", "")
OPENAI_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = "gpt-6-luna"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_PRICE_PER_INPUT_TOKEN = 0.10 / 1_000_000
OPENAI_PRICE_PER_OUTPUT_TOKEN = 0.50 / 1_000_000
