from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator.tools import ToolValidationError, get_tool, list_execution_tools, validate_tool_args
from src.savestate_git import rollback, snapshot
from src.tool_onboarding import init_tool, load_registry, load_tool_spec

ACTION_CLASSES = {"READ", "WRITE", "NETWORK", "SYSTEM"}


class BridgeValidationError(ValueError):
    pass


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


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
    built_in = list_execution_tools()
    built_in_names = {item.get("name") for item in built_in if isinstance(item, dict)}

    merged = list(built_in)
    registry = load_registry()
    for entry in registry.get("tools", []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip() or name in built_in_names:
            continue
        merged.append(
            {
                "name": name,
                "version": entry.get("version", "unknown"),
                "description": entry.get("description")
                or f"Onboarded CLI tool ({entry.get('path', 'unknown path')})",
                "action_class": "SYSTEM",
                "args_schema": {"type": "object"},
                "usage": entry.get("usage", f"{name} [options]"),
            }
        )
    merged.append(
        {
            "name": "tool_onboard",
            "version": "1.0.0",
            "description": "Onboard a CLI tool by reading '<tool> -h' and generating local JSON cheat sheets.",
            "action_class": "SYSTEM",
            "args_schema": {
                "type": "object",
                "required": ["tool_name"],
                "properties": {"tool_name": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
            "usage": "tool_onboard --tool_name <tool>",
        }
    )
    return merged


def _onboarded_tool_entry(tool_name: str) -> dict | None:
    registry = load_registry()
    for entry in registry.get("tools", []):
        if isinstance(entry, dict) and entry.get("name") == tool_name:
            return entry
    return None


def _norm_option_key(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _resolve_flag(spec: dict, arg_key: str) -> str:
    options = spec.get("options")
    if not isinstance(options, dict):
        return f"--{arg_key.replace('_', '-')}"

    wanted = _norm_option_key(arg_key)
    direct = options.get(arg_key)
    if isinstance(direct, dict):
        flags = direct.get("flags")
        if isinstance(flags, list) and flags:
            long_flag = next((f for f in flags if isinstance(f, str) and f.startswith("--")), None)
            return long_flag or str(flags[0])

    for key, option in options.items():
        if _norm_option_key(key) == wanted and isinstance(option, dict):
            flags = option.get("flags")
            if isinstance(flags, list) and flags:
                long_flag = next((f for f in flags if isinstance(f, str) and f.startswith("--")), None)
                return long_flag or str(flags[0])
    return f"--{arg_key.replace('_', '-')}"


def _has_option_key(spec: dict, arg_key: str) -> bool:
    options = spec.get("options")
    if not isinstance(options, dict):
        return False
    wanted = _norm_option_key(arg_key)
    if isinstance(options.get(arg_key), dict):
        return True
    return any(_norm_option_key(key) == wanted for key in options.keys())


def _build_onboarded_argv(tool_name: str, tool_path: str, spec: dict, args: dict) -> list[str]:
    argv: list[str] = [tool_path or tool_name]
    positional = list(args.get("_positional", [])) if isinstance(args.get("_positional"), list) else []
    positional_keys = {"target", "targets", "host", "hosts", "ip", "address", "destination"}
    scan_type_keys = {"scan_type", "scan", "mode", "type"}
    scan_type_ignored_values = {"host", "basic", "default"}
    nmap_scan_type_flag = {
        "ping": "-sn",
        "ping_scan": "-sn",
        "service": "-sV",
        "service_scan": "-sV",
        "syn": "-sS",
        "syn_scan": "-sS",
    }

    def _append_positional(value) -> None:
        if value is None:
            return
        if isinstance(value, list):
            for item in value:
                _append_positional(item)
            return
        if isinstance(value, bool):
            if value:
                positional.append("true")
            return
        text = str(value).strip()
        if not text:
            return
        if " " in text and any(ch in text for ch in ("-", ".", "/", ":")):
            try:
                parts = shlex.split(text)
                if parts:
                    positional.extend(parts)
                    return
            except Exception:
                pass
        positional.append(text)

    for key, value in args.items():
        if key == "_positional":
            continue
        if key in positional_keys:
            _append_positional(value)
            continue

        key_known = _has_option_key(spec, key)
        if not key_known:
            if tool_name.lower() == "nmap" and key in scan_type_keys:
                normalized = str(value).strip().lower()
                mapped_flag = nmap_scan_type_flag.get(normalized)
                if mapped_flag:
                    argv.append(mapped_flag)
                    continue
            if key in scan_type_keys and str(value).strip().lower() in scan_type_ignored_values:
                continue
            _append_positional(value)
            continue

        flag = _resolve_flag(spec, key)
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                argv.extend([flag, str(item)])
            continue
        if value is None:
            continue
        argv.extend([flag, str(value)])

    if isinstance(positional, list):
        for item in positional:
            argv.append(str(item))
    return argv


def _dispatch_tool(call: dict) -> dict:
    validated = _validate_call(call)
    if validated["tool_name"] == "tool_onboard":
        tool_name = str(validated["args"].get("tool_name", "")).strip()
        if not tool_name:
            return _error_result("missing required args: tool_name", exit_code=2)
        try:
            onboard = init_tool(tool_name)
            return {
                "ok": True,
                "exit_code": 0,
                "stdout": json.dumps({"tool_name": tool_name, **onboard}, ensure_ascii=False),
                "stderr": "",
                "duration_ms": 0,
                "artifacts": [],
                "command_preview": f"tool_onboard {shlex.quote(tool_name)}",
            }
        except Exception as exc:
            return _error_result(f"tool onboarding failed: {exc}", exit_code=2)

    tool = None
    argv: list[str]
    tool_name = validated["tool_name"]
    display_name = tool_name

    try:
        tool = get_tool(tool_name)
    except ToolValidationError:
        tool = None

    if tool is not None:
        if not tool.enabled_for_execution:
            return _error_result(f"tool disabled for execution: {tool.name}", exit_code=2)
        try:
            validate_tool_args(tool.name, validated["args"])
        except ToolValidationError as exc:
            return _error_result(str(exc), exit_code=2)
        argv = tool.command_builder(validated["args"])
        display_name = tool.name
    else:
        entry = _onboarded_tool_entry(tool_name)
        if entry is None:
            return _error_result(f"unknown tool: {tool_name}", exit_code=2)
        try:
            spec = load_tool_spec(tool_name)
        except Exception as exc:
            return _error_result(f"tool spec not found: {tool_name} ({exc})", exit_code=2)
        tool_path = str(entry.get("path") or tool_name)
        argv = _build_onboarded_argv(tool_name, tool_path, spec, validated["args"])

    timeout_seconds = validated["timeout_seconds"]
    timeout = int(timeout_seconds) if timeout_seconds is not None else 60

    run_env = os.environ.copy()
    if validated["env"]:
        run_env.update(validated["env"])

    run_cwd = Path(validated["cwd"]).expanduser().resolve() if validated["cwd"] else PROJECT_ROOT
    command_preview = " ".join(shlex.quote(part) for part in argv)

    checkpoint_hash = None
    if validated["action_class"] in {"WRITE", "NETWORK", "SYSTEM"}:
        checkpoint_hash = snapshot(f"before {display_name}")

    def _run_process(command: list[str]) -> tuple[int, str, str]:
        try:
            proc = subprocess.run(
                command,
                cwd=str(run_cwd),
                env=run_env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return proc.returncode, _to_text(proc.stdout), _to_text(proc.stderr)
        except subprocess.TimeoutExpired as exc:
            return -9, _to_text(exc.stdout), _to_text(exc.stderr) + f"\nTimed out after {timeout}s"

    started = time.perf_counter()
    exit_code, stdout, stderr = _run_process(argv)

    # If execution failed due permission constraints, retry via su.
    # Using `su -c` is a non-interactive equivalent of entering root shell,
    # running one command, then exiting.
    if exit_code != 0 and "permission denied" in stderr.lower():
        su_cmd = ["su", "-c", command_preview]
        su_exit_code, su_stdout, su_stderr = _run_process(su_cmd)
        if su_exit_code == 0:
            exit_code, stdout, stderr = su_exit_code, su_stdout, su_stderr
            command_preview = " ".join(shlex.quote(part) for part in su_cmd)
        else:
            stderr = (stderr + f"\nSU retry failed: {su_stderr}").strip()
    duration_ms = int((time.perf_counter() - started) * 1000)

    if checkpoint_hash and exit_code != 0:
        try:
            rollback(checkpoint_hash)
            stderr = (_to_text(stderr) + "\nRollback applied.").strip()
        except Exception as exc:
            stderr = (_to_text(stderr) + f"\nRollback failed: {exc}").strip()

    return {
        "ok": exit_code == 0,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "duration_ms": duration_ms,
        "artifacts": [],
        "command_preview": command_preview,
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
