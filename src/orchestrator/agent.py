from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.orchestrator.executor import execute_plan, write_json, write_plan_and_meta
from src.orchestrator.llm import LLMError, extract_json_object, run_llm_text
from src.orchestrator.planner import validate_plan


class AgentLoopError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _analysis_prompt(original_prompt: str, plan: dict, execution: dict, history: list[dict]) -> str:
    payload = {
        "task": "Analyze the execution outcome and produce JSON only.",
        "requirements": [
            "Return ONLY one JSON object.",
            "No markdown, no code fences, no explanations outside JSON.",
            "Object keys must be: summary, observations, errors, confidence.",
            "confidence must be numeric in range 0..1.",
            "If analysis is uncertain, include reasons in errors and lower confidence.",
        ],
        "original_prompt": original_prompt,
        "plan": plan,
        "execution": execution,
        "history": history,
    }
    return json.dumps(payload, ensure_ascii=False)


def _decision_prompt(original_prompt: str, analysis: dict, history: list[dict]) -> str:
    payload = {
        "task": "Decide whether to finish or continue with another plan. Return JSON only.",
        "requirements": [
            "Return ONLY one JSON object.",
            "No markdown, no code fences, no explanations outside JSON.",
            "action must be exactly 'finish' or 'continue'.",
            "reason must be a plain string (not object, not null, not array).",
            "If action is 'continue', include next_plan object.",
            "next_plan must contain only intent fields for each step: id, tool, args, optional timeout_sec.",
        ],
        "original_prompt": original_prompt,
        "analysis": analysis,
        "history": history,
    }
    return json.dumps(payload, ensure_ascii=False)


def validate_analysis(analysis: dict) -> None:
    if not isinstance(analysis, dict):
        raise AgentLoopError("analysis must be object")
    for key in ("summary", "observations", "errors", "confidence"):
        if key not in analysis:
            raise AgentLoopError(f"analysis missing field: {key}")
    if not isinstance(analysis["summary"], str):
        raise AgentLoopError("analysis.summary must be string")
    if not isinstance(analysis["observations"], list) or any(not isinstance(x, str) for x in analysis["observations"]):
        raise AgentLoopError("analysis.observations must be string[]")
    if not isinstance(analysis["errors"], list) or any(not isinstance(x, str) for x in analysis["errors"]):
        raise AgentLoopError("analysis.errors must be string[]")
    confidence = analysis["confidence"]
    if not isinstance(confidence, (int, float)):
        raise AgentLoopError("analysis.confidence must be number")
    if confidence < 0 or confidence > 1:
        raise AgentLoopError("analysis.confidence must be in range 0..1")


def validate_decision(decision: dict) -> None:
    if not isinstance(decision, dict):
        raise AgentLoopError("decision must be object")
    action = decision.get("action")
    if action not in {"finish", "continue"}:
        raise AgentLoopError("decision.action must be 'finish' or 'continue'")
    if action == "continue" and "next_plan" not in decision:
        raise AgentLoopError("decision.next_plan is required when action=continue")


def _normalize_decision_reason(decision: dict) -> tuple[dict, str | None]:
    if not isinstance(decision, dict):
        return decision, "decision is not an object"

    if "reason" not in decision:
        decision["reason"] = ""
        return decision, "decision.reason missing; defaulted to empty string"

    reason = decision.get("reason")
    if isinstance(reason, str):
        return decision, None

    if isinstance(reason, (dict, list)):
        try:
            decision["reason"] = json.dumps(reason, ensure_ascii=False)
        except Exception:
            decision["reason"] = str(reason)
    else:
        decision["reason"] = "" if reason is None else str(reason)

    return decision, f"decision.reason coerced from {type(reason).__name__} to string"


def _next_plan_from_step_like(step_obj: dict, iteration: int) -> dict:
    step = dict(step_obj)
    if "optional_timeout_sec" in step and "timeout_sec" not in step:
        step["timeout_sec"] = step.pop("optional_timeout_sec")
    else:
        step.pop("optional_timeout_sec", None)

    if "id" not in step or not isinstance(step.get("id"), str) or not step.get("id", "").strip():
        step["id"] = f"step_{iteration:03d}_1"

    return {
        "plan_id": f"agent_plan_{iteration:03d}",
        "session_name": "agent_loop",
        "steps": [step],
        "final_output": "final.json",
    }


def _normalize_next_plan(next_plan_value, iteration: int) -> tuple[dict, str | None]:
    if not isinstance(next_plan_value, dict):
        raise AgentLoopError("next_plan is not an object")

    # Case 1: full plan object
    if "steps" in next_plan_value or "plan_id" in next_plan_value or "final_output" in next_plan_value:
        plan = dict(next_plan_value)
        note_parts: list[str] = []
        steps = plan.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict) and "optional_timeout_sec" in step and "timeout_sec" not in step:
                    step["timeout_sec"] = step.pop("optional_timeout_sec")
                    note_parts.append("mapped optional_timeout_sec to timeout_sec in step")
                elif isinstance(step, dict):
                    step.pop("optional_timeout_sec", None)
        return plan, "; ".join(note_parts) if note_parts else None

    # Case 2: step-like object at top-level
    has_tool_args = "tool" in next_plan_value and "args" in next_plan_value
    has_id_tool_args = {"id", "tool", "args"}.issubset(next_plan_value.keys())
    if has_tool_args or has_id_tool_args:
        plan = _next_plan_from_step_like(next_plan_value, iteration)
        note = "wrapped step-like next_plan into full plan"
        if "optional_timeout_sec" in next_plan_value:
            note += "; mapped optional_timeout_sec to timeout_sec"
        return plan, note

    raise AgentLoopError("next_plan is neither full plan nor step-like object")


def _fallback_analysis(error: str) -> dict:
    return {
        "summary": "analysis_failed",
        "observations": [],
        "errors": [error],
        "confidence": 0.0,
    }


def _fallback_decision(error: str) -> dict:
    return {
        "action": "finish",
        "reason": f"decision_failed: {error}",
    }


def _generate_analysis(
    original_prompt: str,
    plan: dict,
    execution: dict,
    history: list[dict],
    *,
    model_path: str,
    n_ctx: int,
    temp: float,
    seed: int,
    log_dir: Path,
) -> dict:
    prompt = _analysis_prompt(original_prompt, plan, execution, history)
    text = run_llm_text(
        prompt,
        Path(__file__).with_name("plan.schema.json"),
        model_path=model_path,
        n_ctx=n_ctx,
        temp=temp,
        seed=seed,
        log_dir=log_dir,
    )
    analysis = extract_json_object(text)
    validate_analysis(analysis)
    return analysis


def _generate_decision(
    original_prompt: str,
    analysis: dict,
    history: list[dict],
    *,
    model_path: str,
    n_ctx: int,
    temp: float,
    seed: int,
    log_dir: Path,
) -> dict:
    prompt = _decision_prompt(original_prompt, analysis, history)
    text = run_llm_text(
        prompt,
        Path(__file__).with_name("plan.schema.json"),
        model_path=model_path,
        n_ctx=n_ctx,
        temp=temp,
        seed=seed,
        log_dir=log_dir,
    )
    decision = extract_json_object(text)
    return decision


def _failed_execution(plan: dict, reason: str) -> dict:
    return {
        "plan_id": plan.get("plan_id", "unknown"),
        "session_name": plan.get("session_name", "session"),
        "success": False,
        "notes": reason,
        "steps_total": len(plan.get("steps", [])) if isinstance(plan.get("steps"), list) else 0,
        "steps_executed": 0,
        "elapsed_sec": 0.0,
        "results": [],
        "completed_at": _utc_now(),
    }


def _plan_signature(plan: dict) -> str:
    slim = {
        "steps": [
            {
                "tool": step.get("tool"),
                "args": step.get("args"),
                "timeout_sec": step.get("timeout_sec"),
            }
            for step in plan.get("steps", [])
            if isinstance(step, dict)
        ]
    }
    return json.dumps(slim, ensure_ascii=False, sort_keys=True)


def run_agent_loop(
    *,
    session_dir: Path,
    original_prompt: str,
    initial_plan: dict,
    meta: dict,
    model_path: str,
    n_ctx: int,
    temp: float,
    seed: int,
    max_steps: int = 5,
) -> dict:
    session_dir.mkdir(parents=True, exist_ok=True)
    iterations_dir = session_dir / "iterations"
    iterations_dir.mkdir(parents=True, exist_ok=True)

    session_log: dict = {
        "original_prompt": original_prompt,
        "started_at": _utc_now(),
        "max_steps": max_steps,
        "iterations": [],
    }

    consecutive_failures = 0
    seen_signatures: set[str] = set()
    plan = initial_plan
    final_summary = "agent_finished"

    for idx in range(1, max_steps + 1):
        iter_dir = iterations_dir / f"iter_{idx:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        try:
            validate_plan(plan, for_execution=True)
            execution = execute_plan(plan, iter_dir)
            write_plan_and_meta(iter_dir, plan, meta)
        except Exception as exc:
            execution = _failed_execution(plan, f"execution_failed: {exc}")
            write_json(iter_dir / "plan.json", plan)
            write_json(iter_dir / "meta.json", meta)
            write_json(iter_dir / "final.json", execution)

        if execution.get("success"):
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        history = [
            {
                "iteration": it["iteration"],
                "plan_id": it.get("plan_id"),
                "success": it.get("execution", {}).get("success"),
                "decision_action": it.get("decision", {}).get("action"),
            }
            for it in session_log["iterations"]
        ]

        try:
            analysis = _generate_analysis(
                original_prompt,
                plan,
                execution,
                history,
                model_path=model_path,
                n_ctx=n_ctx,
                temp=temp,
                seed=seed,
                log_dir=iter_dir / "llm_analysis",
            )
        except Exception as exc:
            analysis = _fallback_analysis(str(exc))

        try:
            decision_raw = _generate_decision(
                original_prompt,
                analysis,
                history,
                model_path=model_path,
                n_ctx=n_ctx,
                temp=temp,
                seed=seed,
                log_dir=iter_dir / "llm_decision",
            )
            decision, reason_note = _normalize_decision_reason(decision_raw)
            validate_decision(decision)
            decision_raw_payload = {"raw_decision": decision_raw}
            if reason_note:
                decision_raw_payload["reason_note"] = reason_note
        except Exception as exc:
            decision_raw = None
            decision = _fallback_decision(str(exc))
            decision_raw_payload = {
                "raw_decision": decision_raw,
                "reason_note": f"decision_failed: {exc}",
            }

        write_json(iter_dir / "analysis.json", analysis)
        write_json(iter_dir / "decision_raw.json", decision_raw_payload)
        write_json(iter_dir / "decision.json", decision)

        sig = _plan_signature(plan)
        repeated_plan = sig in seen_signatures
        seen_signatures.add(sig)

        iteration_entry = {
            "iteration": idx,
            "plan_id": plan.get("plan_id"),
            "plan": plan,
            "execution": execution,
            "analysis": analysis,
            "decision": decision,
            "timestamp": _utc_now(),
        }
        session_log["iterations"].append(iteration_entry)
        session_log["last_iteration"] = idx
        write_json(session_dir / "session.json", session_log)

        if repeated_plan:
            final_summary = "stopped: identical plan repetition"
            break
        if consecutive_failures >= 2:
            final_summary = "stopped: repeated failures"
            break

        if decision.get("action") == "finish":
            final_summary = decision.get("reason", "finished")
            break

        if decision.get("action") != "continue":
            final_summary = "stopped: invalid decision action"
            break

        raw_next_plan = decision.get("next_plan")
        write_json(iter_dir / "next_plan_raw.json", {"next_plan": raw_next_plan})

        try:
            next_plan, normalize_note = _normalize_next_plan(raw_next_plan, idx)
            if normalize_note:
                decision_raw_payload["next_plan_note"] = normalize_note
                write_json(iter_dir / "decision_raw.json", decision_raw_payload)
            validate_plan(next_plan, for_execution=True)
        except Exception as exc:
            final_summary = f"stopped: next_plan invalid ({exc})"
            break

        plan = next_plan
    else:
        final_summary = "stopped: max_steps reached"

    session_log["finished_at"] = _utc_now()
    session_log["summary"] = final_summary
    write_json(session_dir / "session.json", session_log)

    return {
        "summary": final_summary,
        "iterations": len(session_log["iterations"]),
        "session_dir": str(session_dir),
    }
