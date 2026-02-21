from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    args_schema: dict
    enabled_for_planning: bool
    enabled_for_execution: bool
    command_builder: Callable[[dict], list[str]]


class ToolValidationError(ValueError):
    pass


def _require_keys(args: dict, required: list[str]) -> None:
    missing = [key for key in required if key not in args]
    if missing:
        raise ToolValidationError(f"missing required args: {', '.join(missing)}")


def _validate_ping_args(args: dict) -> None:
    if not isinstance(args, dict):
        raise ToolValidationError("args must be an object")
    _require_keys(args, ["target"])
    target = args["target"]
    if not isinstance(target, str) or not target.strip():
        raise ToolValidationError("target must be a non-empty string")
    count = args.get("count", 4)
    if not isinstance(count, int) or count < 1 or count > 10:
        raise ToolValidationError("count must be an integer between 1 and 10")


def _validate_smartar_args(args: dict) -> None:
    if not isinstance(args, dict):
        raise ToolValidationError("args must be an object")
    _require_keys(args, ["db", "archive", "query"])
    for field in ("db", "archive", "query"):
        value = args[field]
        if not isinstance(value, str) or not value.strip():
            raise ToolValidationError(f"{field} must be a non-empty string")


def _ping_cmd(args: dict) -> list[str]:
    _validate_ping_args(args)
    script = PROJECT_ROOT / "tools" / "ping" / "ping.sh"
    return ["bash", str(script), args["target"], str(args.get("count", 4))]


def _smartar_cmd(args: dict) -> list[str]:
    _validate_smartar_args(args)
    script = PROJECT_ROOT / "tools" / "SmarTar" / "SmarTar" / "run.sh"
    return ["bash", str(script), args["db"], args["archive"], args["query"]]


REGISTRY: dict[str, ToolDef] = {
    "ping": ToolDef(
        name="ping",
        description="Check reachability/latency to a host.",
        args_schema={
            "type": "object",
            "required": ["target"],
            "properties": {
                "target": {"type": "string", "minLength": 1},
                "count": {"type": "integer", "minimum": 1, "maximum": 10, "default": 4},
            },
            "additionalProperties": False,
        },
        enabled_for_planning=True,
        enabled_for_execution=True,
        command_builder=_ping_cmd,
    ),
    "SmarTar": ToolDef(
        name="SmarTar",
        description="Search archive index and extract best match.",
        args_schema={
            "type": "object",
            "required": ["db", "archive", "query"],
            "properties": {
                "db": {"type": "string", "minLength": 1},
                "archive": {"type": "string", "minLength": 1},
                "query": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        enabled_for_planning=False,
        enabled_for_execution=False,
        command_builder=_smartar_cmd,
    ),
}


def get_tool(name: str) -> ToolDef:
    tool = REGISTRY.get(name)
    if tool is None:
        raise ToolValidationError(f"unknown tool: {name}")
    return tool


def list_planning_tools() -> list[dict]:
    tools: list[dict] = []
    for tool in REGISTRY.values():
        if tool.enabled_for_planning:
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "args_schema": tool.args_schema,
                }
            )
    return tools


def validate_tool_args(tool_name: str, args: dict) -> None:
    if tool_name == "ping":
        _validate_ping_args(args)
        return
    if tool_name == "SmarTar":
        _validate_smartar_args(args)
        return
    raise ToolValidationError(f"unknown tool: {tool_name}")
