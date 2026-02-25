# Architecture

## Step 0 Discovery Summary

- OpenClaw reference code is present under `openclaw/`.
- Relevant reference concepts extracted (not reused as runtime):
  - Tool registry/catalog pattern from `openclaw/src/gateway/server-methods/tools-catalog.ts`.
  - Reasoning -> action dispatch flow from `openclaw/src/gateway/server-methods/chat.ts`.
- LOA operator/execution layer entrypoints in this repo:
  - CLI: `src/loa.py`
  - Tool registry: `src/orchestrator/tools.py`
  - Execution dispatcher: `src/orchestrator/executor.py`

## Runtime Split

- Assistant layer (OpenClaw-inspired, lightweight):
  - `src/assistant_core.py`
  - Responsibilities:
    - Receives user input.
    - Loads tool registry dynamically from bridge (`--list-tools`).
    - Uses LLM to choose `respond` vs `tool` call.
    - Sends structured call to LOA bridge.
    - Feeds tool result back to LLM for final answer.

- Operator/execution layer (LOA):
  - Existing tool registry and command builders in `src/orchestrator/tools.py`.
  - Bridge adapter in `src/loa_bridge.py` + `bin/loa-bridge`.
  - Bridge contract:
    - stdin: `protocol/loa_call.schema.json`
    - stdout: `protocol/loa_result.schema.json`

## Deterministic Protocol

- Request fields:
  - `tool_name`, `args`, `cwd`, `timeout_seconds`, `action_class`, `env`
- Result fields:
  - `ok`, `exit_code`, `stdout`, `stderr`, `duration_ms`, `artifacts`
- Design notes:
  - Strict field checking (`additionalProperties: false` in schema).
  - Non-interactive JSON-only bridge.
  - Tool list is dynamic and sourced from LOA registry.

## Save-State and Rollback

- Module: `src/savestate_git.py`
- Snapshot:
  - `git add -A`
  - `git commit -m "checkpoint: <label>"` (only if staged changes exist)
  - returns `HEAD` commit hash
- Rollback:
  - `git reset --hard <commit_hash>`
  - `git clean -fd -e .memory -e .secrets -e .logs`
- Integration in bridge:
  - Pre-exec snapshot for `WRITE`, `NETWORK`, `SYSTEM`
  - Auto-rollback when exit code is non-zero

## Termux Constraints Applied

- No dependency on systemd/docker/GUI.
- Python stdlib-based implementation.
- CLI-first process model.
- No absolute `/usr/bin` assumptions.
- Subprocess execution via shell-available commands.

