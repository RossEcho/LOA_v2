from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator.agent import run_agent_loop
from src.orchestrator.executor import build_meta, create_session_dir, execute_plan, write_plan_and_meta
from src.orchestrator.planner import generate_plan, sanitize_session_name, validate_plan


def _config_path() -> Path:
    raw = os.getenv("LOA_CONFIG_PATH", "~/.loa/config.json")
    return Path(os.path.expanduser(raw)).resolve()


def _load_user_config() -> dict:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_user_config(payload: dict) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _apply_config_to_env() -> None:
    cfg = _load_user_config()
    mapping = {
        "loa_model_path": "LOA_MODEL_PATH",
        "archive_ai_rerank_model": "ARCHIVE_AI_RERANK_MODEL",
        "archive_ai_embed_model": "ARCHIVE_AI_EMBED_MODEL",
    }
    for key, env_name in mapping.items():
        value = cfg.get(key)
        if value and not os.getenv(env_name):
            os.environ[env_name] = str(value)


def _resolve_model_path() -> str:
    cfg = _load_user_config()
    value = (
        os.getenv("LOA_MODEL_PATH")
        or cfg.get("loa_model_path")
        or os.getenv("ARCHIVE_AI_RERANK_MODEL")
        or cfg.get("archive_ai_rerank_model")
    )
    if not value:
        raise RuntimeError("LOA_MODEL_PATH or ARCHIVE_AI_RERANK_MODEL must be set")
    return os.path.expanduser(value)


def _configure_models_interactive() -> None:
    cfg = _load_user_config()
    current_loa = os.getenv("LOA_MODEL_PATH") or cfg.get("loa_model_path", "")
    current_rerank = os.getenv("ARCHIVE_AI_RERANK_MODEL") or cfg.get("archive_ai_rerank_model", "")
    current_embed = os.getenv("ARCHIVE_AI_EMBED_MODEL") or cfg.get("archive_ai_embed_model", "")

    print("\nModel Path Configuration")
    print(f"Config file: {_config_path()}")
    print(f"Current LOA model: {current_loa or '(unset)'}")
    print(f"Current rerank model: {current_rerank or '(unset)'}")
    print(f"Current embed model: {current_embed or '(unset)'}")

    loa_in = input("Set LOA model path (blank keeps current): ").strip()
    rerank_in = input("Set rerank model path (blank keeps current): ").strip()
    embed_in = input("Set embed model path (blank keeps current): ").strip()

    if loa_in:
        cfg["loa_model_path"] = os.path.expanduser(loa_in)
        os.environ["LOA_MODEL_PATH"] = cfg["loa_model_path"]
    elif current_loa:
        cfg["loa_model_path"] = current_loa

    if rerank_in:
        cfg["archive_ai_rerank_model"] = os.path.expanduser(rerank_in)
        os.environ["ARCHIVE_AI_RERANK_MODEL"] = cfg["archive_ai_rerank_model"]
    elif current_rerank:
        cfg["archive_ai_rerank_model"] = current_rerank

    if embed_in:
        cfg["archive_ai_embed_model"] = os.path.expanduser(embed_in)
        os.environ["ARCHIVE_AI_EMBED_MODEL"] = cfg["archive_ai_embed_model"]
    elif current_embed:
        cfg["archive_ai_embed_model"] = current_embed

    _save_user_config(cfg)
    print("Model paths saved")


def _session_from_prompt(prompt: str, yes: bool) -> int:
    model_path = _resolve_model_path()
    n_ctx = int(os.getenv("LOA_N_CTX", "2048"))
    temp = float(os.getenv("LOA_TEMP", "0"))
    seed = int(os.getenv("LOA_SEED", "0"))

    session_seed = sanitize_session_name((prompt or "session")[:80])
    session_dir = create_session_dir(session_seed)

    plan = generate_plan(
        prompt,
        model_path=model_path,
        n_ctx=n_ctx,
        temp=temp,
        seed=seed,
        llm_log_dir=session_dir,
    )
    validate_plan(plan, for_execution=False)
    meta = build_meta(model_path=model_path, n_ctx=n_ctx, temp=temp, seed=seed)
    write_plan_and_meta(session_dir, plan, meta)

    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if yes is None:
        return 0

    if not yes:
        reply = input(f"Run this plan now? [y/N] session={session_dir}: ").strip().lower()
        yes = reply in {"y", "yes"}

    if yes:
        result = execute_plan(plan, session_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0


def _run_session(session_dir_arg: str, yes: bool) -> int:
    session_dir = Path(session_dir_arg).expanduser().resolve()
    plan_path = session_dir / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"plan.json not found in session: {session_dir}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan, for_execution=True)

    if not yes:
        reply = input(f"Execute plan in {session_dir}? [y/N]: ").strip().lower()
        yes = reply in {"y", "yes"}
    if not yes:
        return 0

    result = execute_plan(plan, session_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _agent_from_prompt(
    prompt: str,
    *,
    max_steps: int,
    multi_step: bool,
    max_retries: int,
    max_expansions: int,
    max_replans: int,
    max_runtime_sec: int | None,
) -> int:
    model_path = _resolve_model_path()
    n_ctx = int(os.getenv("LOA_N_CTX", "2048"))
    seed = int(os.getenv("LOA_SEED", "0"))
    temp = 0.0

    session_seed = sanitize_session_name((prompt or "session")[:80])
    session_dir = create_session_dir(session_seed)

    initial_plan = generate_plan(
        prompt,
        model_path=model_path,
        n_ctx=n_ctx,
        temp=temp,
        seed=seed,
        llm_log_dir=session_dir / "llm_plan_initial",
    )
    validate_plan(initial_plan, for_execution=False)

    meta = build_meta(model_path=model_path, n_ctx=n_ctx, temp=temp, seed=seed)
    summary = run_agent_loop(
        session_dir=session_dir,
        original_prompt=prompt,
        initial_plan=initial_plan,
        meta=meta,
        model_path=model_path,
        n_ctx=n_ctx,
        temp=temp,
        seed=seed,
        max_steps=max_steps,
        multi_step_mode=multi_step,
        max_retries=max_retries,
        max_expansions=max_expansions,
        max_replans=max_replans,
        max_runtime_sec=max_runtime_sec,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _interactive_menu() -> int:
    while True:
        print("\nLOA Menu")
        print("1) Plan only")
        print("2) Run existing session")
        print("3) Ask (plan + optional run)")
        print("4) Configure model paths")
        print("5) Agent loop")
        print("6) Exit")
        choice = input("Select [1-6]: ").strip()

        if choice == "1":
            prompt = input("Prompt: ").strip()
            if not prompt:
                print("Prompt cannot be empty")
                continue
            _session_from_prompt(prompt, yes=None)
            continue

        if choice == "2":
            session_dir = input("Session dir path: ").strip()
            if not session_dir:
                print("Session dir path cannot be empty")
                continue
            yes_raw = input("Skip approval prompt? [y/N]: ").strip().lower()
            _run_session(session_dir, yes=yes_raw in {"y", "yes"})
            continue

        if choice == "3":
            prompt = input("Prompt: ").strip()
            if not prompt:
                print("Prompt cannot be empty")
                continue
            yes_raw = input("Auto-approve execution? [y/N]: ").strip().lower()
            _session_from_prompt(prompt, yes=yes_raw in {"y", "yes"})
            continue

        if choice == "4":
            _configure_models_interactive()
            continue

        if choice == "5":
            prompt = input("Prompt: ").strip()
            if not prompt:
                print("Prompt cannot be empty")
                continue
            max_steps_raw = input("Max steps [default 5]: ").strip()
            if not max_steps_raw:
                max_steps = 5
            else:
                try:
                    max_steps = max(1, int(max_steps_raw))
                except ValueError:
                    print("Max steps must be an integer")
                    continue
            multi_step_raw = input("Enable multi-step mode? [y/N]: ").strip().lower()
            multi_step = multi_step_raw in {"y", "yes"}
            _agent_from_prompt(
                prompt,
                max_steps=max_steps,
                multi_step=multi_step,
                max_retries=1,
                max_expansions=2,
                max_replans=1,
                max_runtime_sec=None,
            )
            continue

        if choice == "6":
            return 0

        print("Invalid selection")


def main() -> int:
    _apply_config_to_env()

    parser = argparse.ArgumentParser(prog="loa", description="Local Orchestrator Agent")
    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("menu", help="Interactive menu")

    plan_cmd = sub.add_parser("plan", help="Plan only")
    plan_cmd.add_argument("prompt", help="User prompt")

    run_cmd = sub.add_parser("run", help="Execute existing session plan")
    run_cmd.add_argument("session_dir", help="Session directory path")
    run_cmd.add_argument("--yes", action="store_true", help="Skip approval prompt")

    ask_cmd = sub.add_parser("ask", help="Plan and optionally run")
    ask_cmd.add_argument("prompt", help="User prompt")
    ask_cmd.add_argument("--yes", action="store_true", help="Skip approval prompt and execute")

    agent_cmd = sub.add_parser("agent", help="Run planner-executor-analyzer-decision loop")
    agent_cmd.add_argument("prompt", help="User prompt")
    agent_cmd.add_argument("--max-steps", type=int, default=5, help="Maximum loop iterations")
    agent_cmd.add_argument("--multi-step", action="store_true", help="Enable sequential multi-step execution mode")
    agent_cmd.add_argument("--max-retries", type=int, default=1, help="Maximum retries per step in multi-step mode")
    agent_cmd.add_argument("--max-expansions", type=int, default=2, help="Maximum continuation appends in multi-step mode")
    agent_cmd.add_argument("--max-replans", type=int, default=1, help="Maximum full replans in multi-step mode")
    agent_cmd.add_argument("--max-runtime-sec", type=int, default=None, help="Optional runtime limit for multi-step mode")

    args = parser.parse_args()

    if args.command in (None, "menu"):
        return _interactive_menu()

    if args.command == "plan":
        return _session_from_prompt(args.prompt, yes=None)
    if args.command == "run":
        return _run_session(args.session_dir, yes=args.yes)
    if args.command == "ask":
        return _session_from_prompt(args.prompt, yes=args.yes)
    if args.command == "agent":
        return _agent_from_prompt(
            args.prompt,
            max_steps=max(1, int(args.max_steps)),
            multi_step=bool(args.multi_step),
            max_retries=max(0, int(args.max_retries)),
            max_expansions=max(0, int(args.max_expansions)),
            max_replans=max(0, int(args.max_replans)),
            max_runtime_sec=(None if args.max_runtime_sec is None else max(1, int(args.max_runtime_sec))),
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
