# Runbook

## 1) Termux Setup

Install minimum dependencies:

```sh
pkg update
pkg install python git bash
```

Optional local LLM backend:

- Existing LOA supports server/CLI backends via environment variables in `README.md`.

## 2) Prepare Workspace

From repo root:

```sh
mkdir -p .memory .secrets .logs
```

These directories are excluded from git and protected from rollback cleanup.

## 3) Verify Bridge

List tools:

```sh
python bin/loa-bridge --list-tools
```

Expected shape:

```json
[{"name":"ping","version":"1.0.0","action_class":"NETWORK","args_schema":{...}}]
```

Execute one bridge call:

```sh
printf '%s' '{"tool_name":"ping","args":{"target":"8.8.8.8","count":1},"cwd":null,"timeout_seconds":10,"action_class":"NETWORK","env":null}' | python bin/loa-bridge
```

## 4) Run Assistant

One-shot:

```sh
python main.py --once "ping 8.8.8.8"
```

Interactive:

```sh
python main.py
```

## 5) Add a New Tool

1. Register tool in `src/orchestrator/tools.py`.
2. Provide:
   - `name`
   - `version`
   - `action_class` (`READ|WRITE|NETWORK|SYSTEM`)
   - `args_schema`
   - `command_builder`
3. Enable execution with `enabled_for_execution=True`.
4. Re-run:

```sh
python bin/loa-bridge --list-tools
```

The assistant will pick it up automatically at startup.

## 6) Debug Failures

- Bridge validation errors:
  - Confirm payload matches `protocol/loa_call.schema.json`.
- Tool failures:
  - Check `stderr` and `exit_code` in bridge JSON.
- Rollback behavior:
  - For `WRITE|NETWORK|SYSTEM`, failed tools trigger rollback via `src/savestate_git.py`.
- LLM formatting failures:
  - Ensure model returns JSON objects (assistant expects strict JSON outputs).

## 7) Test Suite

Run all tests:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest
```

Focused tests:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_loa_bridge tests.test_savestate_git tests.test_assistant_core
```

