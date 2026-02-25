from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERSISTED_DIR_EXCLUDES = (".memory", ".secrets", ".logs")


class SaveStateError(RuntimeError):
    pass


def _run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git"] + args,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise SaveStateError(f"git {' '.join(args)} failed: {msg}")
    return proc


def _head_commit() -> str:
    proc = _run_git(["rev-parse", "HEAD"])
    return proc.stdout.strip()


def snapshot(label: str) -> str:
    _run_git(["add", "-A"])
    has_changes = _run_git(["diff", "--cached", "--quiet"], check=False).returncode != 0
    if has_changes:
        _run_git(["commit", "-m", f"checkpoint: {label}"])
    return _head_commit()


def rollback(commit_hash: str) -> None:
    _run_git(["reset", "--hard", commit_hash])
    clean_args = ["clean", "-fd"]
    for excluded in PERSISTED_DIR_EXCLUDES:
        clean_args.extend(["-e", excluded])
    _run_git(clean_args)


def diff(a: str, b: str) -> str:
    proc = _run_git(["diff", a, b], check=False)
    return proc.stdout
