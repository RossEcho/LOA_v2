from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from src.orchestrator.planner import validate_plan
from src.orchestrator.tools import PROJECT_ROOT, get_tool


class ExecutionError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session_dir(session_name: str, runs_root: Path | None = None) -> Path:
    root = runs_root or (PROJECT_ROOT / "runs")
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"{timestamp}__{session_name}"
    suffix = 1
    while path.exists():
        suffix += 1
        path = root / f"{timestamp}__{session_name}_{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _git_hash() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def build_meta(*, model_path: str, n_ctx: int, temp: float, seed: int) -> dict:
    return {
        "created_at": utc_now_iso(),
        "project_root": str(PROJECT_ROOT),
        "model_path": model_path,
        "n_ctx": n_ctx,
        "temp": temp,
        "seed": seed,
        "git_hash": _git_hash(),
        "env": {
            "LOA_LLAMA_CLI_BIN": os.getenv("LOA_LLAMA_CLI_BIN"),
            "LOA_MODEL_PATH": os.getenv("LOA_MODEL_PATH"),
            "ARCHIVE_AI_RERANK_MODEL": os.getenv("ARCHIVE_AI_RERANK_MODEL"),
        },
    }


def write_plan_and_meta(session_dir: Path, plan: dict, meta: dict) -> None:
    write_json(session_dir / "plan.json", plan)
    write_json(session_dir / "meta.json", meta)


def execute_plan(plan: dict, session_dir: Path) -> dict:
    validate_plan(plan, for_execution=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    steps_root = session_dir / "steps"
    steps_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    started = time.perf_counter()

    for idx, step in enumerate(plan["steps"], start=1):
        tool = get_tool(step["tool"])
        if not tool.enabled_for_execution:
            raise ExecutionError(f"tool disabled for execution: {tool.name}")

        step_dir = steps_root / f"step_{idx:03d}_{tool.name}"
        step_dir.mkdir(parents=True, exist_ok=False)

        stdin_text = json.dumps(step["args"], ensure_ascii=False, indent=2)
        (step_dir / "stdin.txt").write_text(stdin_text, encoding="utf-8")

        argv = tool.command_builder(step["args"])
        timeout_sec = int(step.get("timeout_sec", 60))

        t0 = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run(
                argv,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            exit_code = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = -9
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\nTimed out after {timeout_sec}s"
        elapsed = time.perf_counter() - t0

        (step_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
        (step_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
        (step_dir / "exit_code.txt").write_text(str(exit_code), encoding="utf-8")
        write_json(
            step_dir / "timing.json",
            {
                "started_at": utc_now_iso(),
                "elapsed_sec": elapsed,
                "timeout_sec": timeout_sec,
                "timed_out": timed_out,
            },
        )

        step_result = {
            "id": step["id"],
            "tool": tool.name,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "elapsed_sec": elapsed,
            "stdout_file": str(step_dir / "stdout.txt"),
            "stderr_file": str(step_dir / "stderr.txt"),
        }
        results.append(step_result)

        if exit_code != 0:
            break

    total_elapsed = time.perf_counter() - started
    has_planner_failure = isinstance(plan.get("notes"), str) and plan.get("notes", "").startswith("planner_failed:")
    success = (
        len(plan["steps"]) > 0
        and len(results) > 0
        and all(item["exit_code"] == 0 for item in results)
        and len(results) == len(plan["steps"])
        and not has_planner_failure
    )
    final = {
        "plan_id": plan["plan_id"],
        "session_name": plan["session_name"],
        "success": success,
        "notes": plan.get("notes"),
        "steps_total": len(plan["steps"]),
        "steps_executed": len(results),
        "elapsed_sec": total_elapsed,
        "results": results,
        "completed_at": utc_now_iso(),
    }

    final_name = plan.get("final_output", "final.json")
    final_path = session_dir / final_name
    if final_path.suffix.lower() == ".txt":
        final_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        write_json(final_path, final)

    if final_path.name != "final.json":
        write_json(session_dir / "final.json", final)

    return final
