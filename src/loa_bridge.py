from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator.tools import ToolValidationError, get_tool, list_execution_tools, validate_tool_args
from src.savestate_git import rollback, snapshot

ACTION_CLASSES = {"READ", "WRITE", "NETWORK", "SYSTEM"}


class BridgeValidationError(ValueError):
    pass


def _error_result(message: str, *, exit_code: int = 2) -> dict:
    return {
        "ok": False,
        "exit_code": exit_code,
        "stdout": "",
        "stderr": message,
        "duration_ms": 0,
        "artifacts": [],
    }


def _validate_call(call: dict) -> dict:
    if not isinstance(call, dict):
        raise BridgeValidationError("request must be a JSON object")

    required = {"tool_name", "args", "cwd", "timeout_seconds", "action_class", "env"}
    extra = set(call.keys()) - required
    if extra:
        raise BridgeValidationError(f"unknown request fields: {', '.join(sorted(extra))}")
    missing = required - set(call.keys())
    if missing:
        raise BridgeValidationError(f"missing request fields: {', '.join(sorted(missing))}")

    tool_name = call["tool_name"]
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise BridgeValidationError("tool_name must be a non-empty string")

    args = call["args"]
    if not isinstance(args, dict):
        raise BridgeValidationError("args must be an object")

    cwd = call["cwd"]
    if cwd is not None and (not isinstance(cwd, str) or not cwd.strip()):
        raise BridgeValidationError("cwd must be null or a non-empty string")

    timeout_seconds = call["timeout_seconds"]
    if timeout_seconds is not None:
        if not isinstance(timeout_seconds, (int, float)):
            raise BridgeValidationError("timeout_seconds must be null or a number")
        if timeout_seconds <= 0:
            raise BridgeValidationError("timeout_seconds must be > 0 when provided")

    action_class = call["action_class"]
    if action_class not in ACTION_CLASSES:
        raise BridgeValidationError("action_class must be READ, WRITE, NETWORK, or SYSTEM")

    env = call["env"]
    if env is not None:
        if not isinstance(env, dict):
            raise BridgeValidationError("env must be null or an object")
        for key, value in env.items():
            if not isinstance(key, str) or not key:
                raise BridgeValidationError("env keys must be non-empty strings")
            if not isinstance(value, str):
                raise BridgeValidationError("env values must be strings")

    return {
        "tool_name": tool_name,
        "args": args,
        "cwd": cwd,
        "timeout_seconds": timeout_seconds,
        "action_class": action_class,
        "env": env,
    }


def _list_tools() -> list[dict]:
    return list_execution_tools()


def _dispatch_tool(call: dict) -> dict:
    validated = _validate_call(call)

    try:
        tool = get_tool(validated["tool_name"])
    except ToolValidationError as exc:
        return _error_result(str(exc), exit_code=2)

    if not tool.enabled_for_execution:
        return _error_result(f"tool disabled for execution: {tool.name}", exit_code=2)

    try:
        validate_tool_args(tool.name, validated["args"])
    except ToolValidationError as exc:
        return _error_result(str(exc), exit_code=2)

    timeout_seconds = validated["timeout_seconds"]
    timeout = int(timeout_seconds) if timeout_seconds is not None else 60

    run_env = os.environ.copy()
    if validated["env"]:
        run_env.update(validated["env"])

    run_cwd = Path(validated["cwd"]).expanduser().resolve() if validated["cwd"] else PROJECT_ROOT
    argv = tool.command_builder(validated["args"])

    checkpoint_hash = None
    if validated["action_class"] in {"WRITE", "NETWORK", "SYSTEM"}:
        checkpoint_hash = snapshot(f"before {tool.name}")

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(run_cwd),
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        exit_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = -9
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTimed out after {timeout}s"
    duration_ms = int((time.perf_counter() - started) * 1000)

    if checkpoint_hash and exit_code != 0:
        try:
            rollback(checkpoint_hash)
            stderr = (stderr + "\nRollback applied.").strip()
        except Exception as exc:
            stderr = (stderr + f"\nRollback failed: {exc}").strip()

    return {
        "ok": exit_code == 0,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "duration_ms": duration_ms,
        "artifacts": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loa-bridge", description="LOA stdin/stdout bridge")
    parser.add_argument("--list-tools", action="store_true", help="Print available executable tools as JSON")
    args = parser.parse_args(argv)

    if args.list_tools:
        print(json.dumps(_list_tools(), ensure_ascii=False))
        return 0

    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print(json.dumps(_error_result("invalid JSON request", exit_code=2), ensure_ascii=False))
        return 2

    try:
        result = _dispatch_tool(payload)
    except BridgeValidationError as exc:
        result = _error_result(str(exc), exit_code=2)
    except Exception as exc:
        result = _error_result(f"bridge internal error: {exc}", exit_code=1)

    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
