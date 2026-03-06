# LOA_v2

LOA is a local orchestrator that plans and executes tool runs in deterministic, archivable sessions.

## What this repo contains

- `src/orchestrator/`: planner, schema validation, tool registry, executor, LLM wrapper.
- `src/loa.py`: CLI entrypoint (`plan`, `run`, `ask`, and interactive `menu`).
- `tools/ping/`: active validation tool used by the orchestrator now.
- `tools/SmarTar/`: registered tool (currently disabled for planning/execution by LOA).

## Runtime target

- Designed for Termux / local environments.
- Keep execution under `$HOME` (no `/sdcard` execution assumptions).

## Model configuration

LOA reads model settings from env first, then from user config:

1. `LOA_MODEL_PATH`
2. `~/.loa/config.json` key: `loa_model_path`
3. `ARCHIVE_AI_RERANK_MODEL`
4. `~/.loa/config.json` key: `archive_ai_rerank_model`

Optional config path override:

- `LOA_CONFIG_PATH` (default `~/.loa/config.json`)

## LLM backend (recommended: llama-server)

Start local server:

```bash
llama-server -m "$LOA_MODEL_PATH" --port 8080
```

LOA defaults to server backend and calls:

- `POST http://127.0.0.1:8080/v1/chat/completions`

Relevant env vars:

- `LOA_LLM_BACKEND=server` (default; set `cli` only if needed)
- `LOA_LLAMA_SERVER_URL` (default `http://127.0.0.1:8080/v1/chat/completions`)
- `LOA_LLAMA_SERVER_MODEL` (default `local`)
- `LOA_LLM_TIMEOUT_SEC` (default `90`)
- `LOA_LLM_MAX_TOKENS` (default `512`)

Deterministic generation settings are passed from orchestrator (`seed`, `temperature`, `max_tokens`).

### Configure via menu

Use menu option **Configure model paths** to persist:

- `loa_model_path`
- `archive_ai_rerank_model`
- `archive_ai_embed_model`

## CLI usage

Run from repo root.

### Unified entrypoint (`main.py`)

Top-level menu:

```bash
python main.py
```

LOA CLI pass-through (example):

```bash
python main.py loa agent "check host connectivity" --multi-step --max-steps 8
```

Assistant mode (example):

```bash
python main.py assistant --once "hello"
```

### Interactive menu

```bash
python src/loa.py
```

or

```bash
python src/loa.py menu
```

### Plan only (creates draft session)

```bash
python src/loa.py plan "ping example.com"
```

### Run existing session

```bash
python src/loa.py run runs/<session_dir>
python src/loa.py run runs/<session_dir> --yes
```

### One-shot ask (plan + approval + optional execute)

```bash
python src/loa.py ask "check host connectivity"
python src/loa.py ask "check host connectivity" --yes
```

### Agent runtime modes

One-shot mode (existing behavior):

```bash
python src/loa.py agent "check host connectivity" --max-steps 5
```

Multi-step mode (sequential execution with continuation + bounded replanning):

```bash
python src/loa.py agent "check host connectivity" --multi-step --max-steps 12 --max-retries 1 --max-expansions 2 --max-replans 1
```

## Session layout

Each run writes to:

`runs/<YYYYMMDD_HHMMSS>__<session_name>/`

Contents:

- `plan.json`: validated plan used for execution.
- `meta.json`: env/model snapshot, seed/temp/context, timestamps, git hash (if available).
- `steps/step_###_<tool>/`
	- `stdin.txt`
	- `stdout.txt`
	- `stderr.txt`
	- `exit_code.txt`
	- `timing.json`
	- optional copied tool output files.
- `final.json` (or plan-defined final file): aggregated execution result.
- `session_state.json` (multi-step mode): live state with current plan, completed/pending/failed steps, tool output summaries, artifacts, appended continuation steps, and replan history.
- `multi_step/` (multi-step mode): per-step atomic executions (`run_###_...`) plus continuation/replan LLM logs.

## Minimal multi-step example

With `--multi-step`, LOA can execute a plan like:

1. `s1`: ping primary host
2. `s2`: ping secondary host (depends on `s1`)
3. continuation append: model adds `s3` if follow-up verification is needed

The runtime executes each step sequentially using the existing atomic executor, validates success criteria, and appends continuation steps without replacing the existing plan unless a full replan is triggered.

## Safety and determinism

- Plans are JSON-only and validated before execution.
- Unknown or disabled tools are refused.
- Tool args are validated against registry rules.
- Execution uses subprocess argv lists (no eval/shell command construction from model output).
- Per-step timeout is enforced.

## Current tool policy

- `ping`: enabled (planning + execution).
- `SmarTar`: registered but disabled in orchestrator flow for now.
