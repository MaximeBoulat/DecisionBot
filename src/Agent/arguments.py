from __future__ import annotations

import json
import re
from dataclasses import dataclass

from Agent.context import DecisionContext


@dataclass(frozen=True)
class ArgumentChoice:
    path: tuple[str, ...]
    schema: dict
    values: dict[str, object]
    criteria: dict[str, str]


def matches(value: object, schema: dict) -> bool:
    kinds = schema.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    valid = (
        (value is None and "null" in kinds)
        or (type(value) is bool and "boolean" in kinds)
        or (type(value) is int and ("integer" in kinds or "number" in kinds))
        or (type(value) is float and "number" in kinds)
        or (isinstance(value, str) and "string" in kinds)
        or (isinstance(value, list) and "array" in kinds)
        or (isinstance(value, dict) and "object" in kinds)
    )
    if valid and "enum" in schema:
        valid = value in schema["enum"]
    if valid and type(value) in (int, float):
        valid = schema.get("minimum", float("-inf")) <= value <= schema.get("maximum", float("inf"))
    if valid and isinstance(value, dict):
        props = schema.get("properties", {})
        valid = set(value) == set(props) and all(matches(value[k], v) for k, v in props.items())
    if valid and isinstance(value, list):
        valid = all(matches(item, schema["items"]) for item in value)
    return valid


def argument_choices(schema: dict, context: DecisionContext, path: tuple[str, ...] = ()) -> list[ArgumentChoice]:
    result = []
    state = context.state
    for name, prop in schema["properties"].items():
        current = path + (name,)
        kind = prop["type"]
        if kind == "object":
            result.extend(argument_choices(prop, context, current))
            continue
        bounded = "enum" in prop or kind == "boolean" or kind == ["boolean", "null"]
        values = list(prop.get("enum", []))
        if "boolean" in kind:
            values.extend([False, True])
        if "null" in kind:
            values.append(None)
        if not bounded:
            if name == "sql":
                pass
            elif "object" in kind or "array" in kind:
                decoder = json.JSONDecoder()
                for match in re.finditer(r"[{\[]", state):
                    try:
                        value, _ = decoder.raw_decode(state[match.start():])
                        values.append(value)
                    except ValueError:
                        pass
            elif "string" in kind:
                values.extend(re.findall(r'"([^"\n]+)"', context.task.question))
                values.extend(re.findall(r"\b[A-Za-z][A-Za-z0-9_.-]*\b", context.task.question))
                values.extend(re.findall(r'"([^"\n]+)"', state))
                values.extend(re.findall(r"\b[A-Za-z][A-Za-z0-9_.-]*\b", state))
            else:
                values.extend(float(s) if "." in s else int(s) for s in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", state))
        unique = {}
        for value in values:
            if matches(value, prop):
                unique.setdefault(json.dumps(value), value)
        selected = list(unique.values())[:200]
        if not selected:
            continue
        options = {f"v{i}": value for i, value in enumerate(selected)}
        criteria = {key: json.dumps(value, ensure_ascii=False) for key, value in options.items()}
        if not bounded:
            criteria["other"] = "<OTHER: none of these / needs generation>"
        if len(criteria) >= 2:
            result.append(ArgumentChoice(current, prop, options, criteria))
    return result


def put_value(target: dict, path: tuple[str, ...], value: object) -> None:
    for name in path[:-1]:
        target = target.setdefault(name, {})
    target[path[-1]] = value


def get_value(target: dict, path: tuple[str, ...]) -> object:
    value = target
    for name in path:
        value = value[name]
    return value


def missing_schema(schema: dict, fixed: dict) -> dict:
    properties = {}
    for name, prop in schema["properties"].items():
        if name not in fixed:
            properties[name] = prop
        elif prop["type"] == "object":
            remaining = missing_schema(prop, fixed[name])
            if remaining["properties"]:
                properties[name] = remaining
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def merge_arguments(fixed: dict, generated: dict) -> dict:
    result = dict(fixed)
    for name, value in generated.items():
        if name in result:
            if not isinstance(value, dict) or not isinstance(result[name], dict):
                raise ValueError(f"Generator attempted to overwrite accepted argument: {name}")
            result[name] = merge_arguments(result[name], value)
        else:
            result[name] = value
    return result
