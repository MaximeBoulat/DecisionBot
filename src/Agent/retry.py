from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable

ATTEMPTS = 4
RETRY_STATUS = {429, 500, 502, 503, 504}


def post_json(url: str, payload: dict, headers: dict[str, str], timeout: float,
              on_retry: Callable[..., None] | None = None) -> dict:
    """POST JSON and return the decoded reply. Timeouts, dropped connections and HTTP 429/5xx are retried
    with waits of 2, 4 and 8 seconds, or the server's Retry-After; other errors are raised at once."""
    data = json.dumps(payload).encode()
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUS or attempt == ATTEMPTS:
                raise
            after = exc.headers.get("Retry-After", "") if exc.headers else ""
            delay = min(int(after), 60) if after.isdigit() else 2 ** attempt
            reason = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == ATTEMPTS:
                raise
            delay, reason = 2 ** attempt, f"{type(exc).__name__}: {exc}"
        print(f" Retrying in {delay}s after {reason} (attempt {attempt + 1}/{ATTEMPTS})...", end="", flush=True)
        if on_retry is not None:
            on_retry(url=url, attempt=attempt + 1, delay_seconds=delay, reason=reason)
        time.sleep(delay)
    raise AssertionError("unreachable")
