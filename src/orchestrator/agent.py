from __future__ import annotations

import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.orchestrator.executor import execute_plan, write_json, write_plan_and_meta
from src.orchestrator.llm import LLMError, extract_json_object, run_llm_text
from src.orchestrator.planner import validate_plan
from src.orchestrator.tools import REGISTRY


class AgentLoopError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collect_output_snapshot(path: Path, max_inline_chars: int = 4000) -> dict:
    snapshot = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists() or not path.is_file():
        return snapshot

    content = path.read_text(encoding="utf-8", errors="replace")
    snapshot["size_chars"] = len(content)
    if len(content) <= max_inline_chars:
        snapshot["inline"] = content
        snapshot["truncated"] = False
    else:
        snapshot["inline"] = content[:max_inline_chars]
        snapshot["truncated"] = True
    return snapshot


def _write_analysis_error_report(
    session_dir: Path,
    iteration: int,
    exc: Exception,
    tb: str,
    llm_output_paths: list[Path],
) -> None:
    outputs = [_collect_output_snapshot(path) for path in llm_output_paths]
    payload = {
        "stage": "analysis",
        "timestamp": _utc_now(),
        "iteration": iteration,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": tb,
        "llm_outputs": outputs,
    }
    write_json(session_dir / "error.json", payload)
    (session_dir / "error.traceback.txt").write_text(tb, encoding="utf-8")


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
        "task": "Choose next agent action. Return JSON only.",
        "requirements": [
            "Return ONLY one JSON object.",
            "No markdown, no code fences, no explanations outside JSON.",
            "Use this schema: {\"decision\":{\"action\":\"run_tool|respond|stop\",\"reason\":\"string\"},\"next_step\":object|null,\"response\":string|null}",
            "If action=run_tool: next_step required, response must be null.",
            "If action=respond: response required, next_step must be null.",
            "If action=stop: both next_step and response must be null.",
            "next_step fields: tool, args, optional id, optional timeout_sec or optional_timeout_sec.",
            "args must be a JSON object; do not put natural language text in args.",
            "reason must be a plain string (not object, not null, not array).",
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
    if action not in {"run_tool", "respond", "stop"}:
        raise AgentLoopError("decision.action must be 'run_tool', 'respond', or 'stop'")


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


def _normalize_step_args(args_value) -> tuple[dict, str | None]:
    if args_value is None:
        return {}, "args missing; defaulted to {}"
    if isinstance(args_value, dict):
        return dict(args_value), None
    if isinstance(args_value, str):
        try:
            parsed = json.loads(args_value)
            if isinstance(parsed, dict):
                return parsed, "args parsed from JSON string"
        except Exception:
            pass
        return {}, "args string not valid JSON object; defaulted to {}"
    return {}, f"args type {type(args_value).__name__} coerced to {{}}"


def _extract_ping_target(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    ip = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    if ip:
        return ip.group(0)
    host = re.search(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b", text)
    if host:
        return host.group(0)
    return None


def _normalize_next_step(raw_step, original_prompt: str, iteration: int) -> tuple[dict, list[str]]:
    if not isinstance(raw_step, dict):
        raise AgentLoopError("next_step must be object for action=run_tool")

    notes: list[str] = []
    step = dict(raw_step)

    if "optional_timeout_sec" in step and "timeout_sec" not in step:
        step["timeout_sec"] = step.pop("optional_timeout_sec")
        notes.append("mapped optional_timeout_sec to timeout_sec")
    else:
        step.pop("optional_timeout_sec", None)

    if "id" not in step or not isinstance(step.get("id"), str) or not step.get("id", "").strip():
        step["id"] = f"step_{iteration:03d}_1"
        notes.append("step id generated")

    args, args_note = _normalize_step_args(step.get("args"))
    if args_note:
        notes.append(args_note)
    step["args"] = args

    tool = step.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        raise AgentLoopError("next_step.tool missing or invalid")
    if tool not in REGISTRY:
        raise AgentLoopError(f"unknown tool: {tool}")

    if tool == "ping":
        target = step["args"].get("target")
        if not isinstance(target, str) or not target.strip():
            recovered = _extract_ping_target(original_prompt)
            if recovered:
                step["args"]["target"] = recovered
                notes.append(f"ping target repaired from prompt: {recovered}")
            else:
                raise AgentLoopError("ping target missing and could not be repaired")

    timeout = step.get("timeout_sec")
    if timeout is not None and not isinstance(timeout, int):
        step.pop("timeout_sec", None)
        notes.append("invalid timeout_sec removed")

    return step, notes


def _normalize_decision_packet(raw_packet, original_prompt: str, iteration: int) -> tuple[dict, dict]:
    if not isinstance(raw_packet, dict):
        raise AgentLoopError("decision packet must be object")

    diagnostic: dict = {"raw_decision": raw_packet}

    decision_obj = raw_packet.get("decision") if isinstance(raw_packet.get("decision"), dict) else raw_packet
    decision_obj = dict(decision_obj) if isinstance(decision_obj, dict) else {}

    if decision_obj.get("action") == "continue":
        decision_obj["action"] = "run_tool"
        diagnostic["action_note"] = "mapped legacy action continue->run_tool"
    if decision_obj.get("action") == "finish":
        decision_obj["action"] = "stop"
        diagnostic["action_note"] = "mapped legacy action finish->stop"

    decision_obj, reason_note = _normalize_decision_reason(decision_obj)
    if reason_note:
        diagnostic["reason_note"] = reason_note

    action = decision_obj.get("action")
    response = raw_packet.get("response")
    raw_next_step = raw_packet.get("next_step")
    if raw_next_step is None:
        raw_next_step = raw_packet.get("next_plan")

    normalized = {
        "decision": decision_obj,
        "next_step": None,
        "response": None,
    }

    if action == "run_tool":
        step, notes = _normalize_next_step(raw_next_step, original_prompt, iteration)
        normalized["next_step"] = step
        normalized["response"] = None
        if notes:
            diagnostic["next_step_notes"] = notes
    elif action == "respond":
        normalized["next_step"] = None
        normalized["response"] = response if isinstance(response, str) else ("" if response is None else str(response))
    elif action == "stop":
        normalized["next_step"] = None
        normalized["response"] = None

    validate_decision(decision_obj)
    return normalized, diagnostic


def _next_plan_from_step(step: dict, iteration: int) -> dict:
    return {
        "plan_id": f"agent_plan_{iteration:03d}",
        "session_name": "agent_loop",
        "steps": [step],
        "final_output": "final.json",
    }


def _fallback_analysis(error: str) -> dict:
    return {
        "summary": "analysis_failed",
        "observations": [],
        "errors": [error],
        "confidence": 0.0,
    }


def _fallback_decision(error: str) -> dict:
    return {
        "action": "stop",
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
            tb = traceback.format_exc()
            _write_analysis_error_report(
                session_dir,
                idx,
                exc,
                tb,
                [
                    iter_dir / "llm_analysis" / "llm_raw_response.json",
                    iter_dir / "llm_analysis" / "llm_text.txt",
                ],
            )
            print(f"analysis failed: {type(exc).__name__}: {exc} (see {session_dir / 'error.json'})")
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
            normalized_packet, decision_raw_payload = _normalize_decision_packet(
                decision_raw,
                original_prompt,
                idx,
            )
            decision = normalized_packet["decision"]
            next_step = normalized_packet["next_step"]
            response = normalized_packet["response"]
            raw_next_step_for_log = None
            if isinstance(decision_raw, dict):
                raw_next_step_for_log = decision_raw.get("next_step")
                if raw_next_step_for_log is None:
                    raw_next_step_for_log = decision_raw.get("next_plan")
        except Exception as exc:
            decision_raw = None
            decision = _fallback_decision(str(exc))
            next_step = None
            response = None
            raw_next_step_for_log = None
            decision_raw_payload = {
                "raw_decision": decision_raw,
                "reason_note": f"decision_failed: {exc}",
            }

        write_json(iter_dir / "analysis.json", analysis)
        write_json(iter_dir / "decision_raw.json", decision_raw_payload)
        write_json(
            iter_dir / "decision.json",
            {
                "decision": decision,
                "next_step": next_step,
                "response": response,
            },
        )
        write_json(
            iter_dir / "next_step_raw.json",
            {
                "raw_next_step": raw_next_step_for_log,
                "normalized_next_step": next_step,
            },
        )

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
            "next_step": next_step,
            "response": response,
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

        if decision.get("action") == "stop":
            final_summary = decision.get("reason", "finished")
            break

        if decision.get("action") == "respond":
            final_summary = response or decision.get("reason", "respond")
            break

        if decision.get("action") != "run_tool":
            final_summary = "stopped: invalid decision action"
            break

        try:
            if not isinstance(next_step, dict):
                raise AgentLoopError("next_step missing for run_tool")
            next_plan = _next_plan_from_step(next_step, idx)
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
