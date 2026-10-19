from __future__ import annotations

from config import OPENAI_PRICE_PER_INPUT_TOKEN, OPENAI_PRICE_PER_OUTPUT_TOKEN


def generator_call_cost(usage: dict | None) -> float:
    if not usage:
        raise ValueError("Missing token usage for a billed generator call")
    return (usage["input_tokens"] * OPENAI_PRICE_PER_INPUT_TOKEN
            + usage["output_tokens"] * OPENAI_PRICE_PER_OUTPUT_TOKEN)
