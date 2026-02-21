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

### Configure via menu

Use menu option **Configure model paths** to persist:

- `loa_model_path`
- `archive_ai_rerank_model`
- `archive_ai_embed_model`

## CLI usage

Run from repo root.

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

## Safety and determinism

- Plans are JSON-only and validated before execution.
- Unknown or disabled tools are refused.
- Tool args are validated against registry rules.
- Execution uses subprocess argv lists (no eval/shell command construction from model output).
- Per-step timeout is enforced.

## Current tool policy

- `ping`: enabled (planning + execution).
- `SmarTar`: registered but disabled in orchestrator flow for now.
