from __future__ import annotations

import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.orchestrator.executor import execute_plan, write_json, write_plan_and_meta
from src.orchestrator.llm import extract_json_object, run_llm_text
from src.orchestrator.planner import generate_plan, validate_plan
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
    if not isinstance(analysis["errors"], list) or any(not isinstance(x, str) for x in analysis["errors"]):
        raise AgentLoopError("analysis.errors must be string[]")
    confidence = analysis["confidence"]
    if not isinstance(confidence, (int, float)):
        raise AgentLoopError("analysis.confidence must be number")
    if confidence < 0 or confidence > 1:
        raise AgentLoopError("analysis.confidence must be in range 0..1")


def _normalize_analysis_payload(analysis: dict) -> tuple[dict, str | None]:
    if not isinstance(analysis, dict):
        raise AgentLoopError("analysis must be object")

    normalized = dict(analysis)
    note = None

    observations = normalized.get("observations")
    if observations is None:
        normalized["observations"] = []
        note = "observations missing/null; defaulted to []"
    elif isinstance(observations, str):
        normalized["observations"] = [observations]
        note = "observations string wrapped into list"
    elif isinstance(observations, list):
        normalized["observations"] = [str(item) for item in observations]
        if any(not isinstance(item, str) for item in observations):
            note = "observations list items coerced to string"
    else:
        normalized["observations"] = [str(observations)]
        note = f"observations coerced from {type(observations).__name__}"

    return normalized, note


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
) -> tuple[dict, dict, str | None]:
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
    raw_analysis = extract_json_object(text)
    analysis, note = _normalize_analysis_payload(raw_analysis)
    validate_analysis(analysis)
    return raw_analysis, analysis, note


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


def _trim_text(value: str, max_chars: int = 1200) -> str:
    if not isinstance(value, str):
        return ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars]


def _collect_step_io(step_result: dict, max_chars: int = 1200) -> dict:
    stdout_path = Path(step_result.get("stdout_file", ""))
    stderr_path = Path(step_result.get("stderr_file", ""))
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    return {
        "stdout": _trim_text(stdout, max_chars=max_chars),
        "stderr": _trim_text(stderr, max_chars=max_chars),
        "stdout_truncated": len(stdout) > max_chars,
        "stderr_truncated": len(stderr) > max_chars,
    }


def _validate_step_result(step: dict, step_result: dict) -> tuple[bool, list[str], dict]:
    criteria = step.get("success_criteria", {"type": "exit_code", "equals": 0})
    io = _collect_step_io(step_result)
    failures: list[str] = []

    expected_code = int(criteria.get("equals", 0))
    actual_code = int(step_result.get("exit_code", -1))
    if actual_code != expected_code:
        failures.append(f"exit_code {actual_code} != {expected_code}")

    for snippet in criteria.get("stdout_contains", []):
        if snippet not in io["stdout"]:
            failures.append(f"stdout missing snippet: {snippet}")

    for snippet in criteria.get("stderr_contains", []):
        if snippet not in io["stderr"]:
            failures.append(f"stderr missing snippet: {snippet}")

    return len(failures) == 0, failures, io


def _step_plan(plan: dict, step: dict, run_index: int, attempt: int) -> dict:
    step_copy = dict(step)
    step_copy.pop("depends_on", None)
    step_copy.pop("success_criteria", None)
    step_copy.pop("retry_policy", None)
    step_copy.pop("description", None)
    return {
        "plan_id": f"{plan.get('plan_id', 'plan')}_run_{run_index:03d}_{step['id']}_try{attempt}",
        "session_name": plan.get("session_name", "session"),
        "steps": [step_copy],
        "final_output": "final.json",
    }


def _sorted_plan_steps(plan: dict) -> list[dict]:
    return [step for step in plan.get("steps", []) if isinstance(step, dict)]


def _find_next_executable_step(plan: dict, completed: set[str], failed: set[str], queued: set[str]) -> dict | None:
    for step in _sorted_plan_steps(plan):
        step_id = step["id"]
        if step_id in completed or step_id in failed:
            continue
        deps = step.get("depends_on", [])
        if any(dep in failed for dep in deps):
            continue
        if all(dep in completed for dep in deps):
            if step_id not in queued:
                return step
    return None


def _coerce_planned_step(raw_step: dict, fallback_prefix: str, index: int) -> dict:
    step = dict(raw_step) if isinstance(raw_step, dict) else {}
    if not isinstance(step.get("id"), str) or not step["id"].strip():
        step["id"] = f"{fallback_prefix}_{index:03d}"
    if not isinstance(step.get("description"), str) or not step["description"].strip():
        tool_name = step.get("tool", "tool")
        step["description"] = f"Execute tool '{tool_name}'"
    if not isinstance(step.get("args"), dict):
        step["args"] = {}
    if not isinstance(step.get("depends_on"), list):
        step["depends_on"] = []
    if not isinstance(step.get("success_criteria"), dict):
        step["success_criteria"] = {"type": "exit_code", "equals": 0}
    if not isinstance(step.get("retry_policy"), dict):
        step["retry_policy"] = {"max_retries": 0}
    return step


def _ensure_unique_step_ids(existing_ids: set[str], step: dict, prefix: str) -> None:
    original = step["id"]
    if original not in existing_ids:
        existing_ids.add(original)
        return
    suffix = 1
    while True:
        candidate = f"{prefix}_{suffix}"
        suffix += 1
        if candidate not in existing_ids:
            step["id"] = candidate
            existing_ids.add(candidate)
            return

def _continuation_prompt(goal: str, plan: dict, state_summary: dict) -> str:
    payload = {
        "task": "Generate continuation steps only when strictly needed.",
        "requirements": [
            "Return JSON only.",
            "Return object with keys: append_steps (array), reason (string).",
            "append_steps must contain only NEW steps; never include already executed steps.",
            "If no continuation needed, return append_steps as empty array.",
            "Each appended step must include: id, description, tool, args.",
            "Optionally include: depends_on, success_criteria, retry_policy, timeout_sec.",
            "Keep appended steps minimal and aligned with original goal.",
        ],
        "goal": goal,
        "current_plan": plan,
        "state_summary": state_summary,
    }
    return json.dumps(payload, ensure_ascii=False)


def _parse_continuation_packet(raw: dict) -> tuple[list[dict], str]:
    if not isinstance(raw, dict):
        raise AgentLoopError("continuation payload must be object")
    append_steps = raw.get("append_steps", [])
    if not isinstance(append_steps, list):
        raise AgentLoopError("append_steps must be array")
    reason = raw.get("reason")
    if not isinstance(reason, str):
        reason = ""
    return append_steps, reason


def _state_summary_for_planning(state: dict, max_outputs: int = 8) -> dict:
    output_items = list(state.get("tool_outputs", {}).items())
    trimmed_outputs = []
    for step_id, payload in output_items[-max_outputs:]:
        if not isinstance(payload, dict):
            continue
        trimmed_outputs.append(
            {
                "step_id": step_id,
                "exit_code": payload.get("exit_code"),
                "timed_out": payload.get("timed_out"),
                "stdout_excerpt": payload.get("stdout_excerpt"),
                "stderr_excerpt": payload.get("stderr_excerpt"),
            }
        )
    return {
        "completed_steps": state.get("completed_steps", []),
        "failed_steps": state.get("failed_steps", []),
        "pending_steps": state.get("pending_steps", []),
        "tool_outputs": trimmed_outputs,
    }


def _generate_continuation_steps(
    *,
    goal: str,
    plan: dict,
    state: dict,
    model_path: str,
    n_ctx: int,
    temp: float,
    seed: int,
    log_dir: Path,
    id_prefix: str,
) -> tuple[list[dict], str]:
    prompt = _continuation_prompt(goal, plan, _state_summary_for_planning(state))
    text = run_llm_text(
        prompt,
        Path(__file__).with_name("continuation.schema.json"),
        model_path=model_path,
        n_ctx=n_ctx,
        temp=temp,
        seed=seed,
        log_dir=log_dir,
    )
    packet = extract_json_object(text)
    raw_steps, reason = _parse_continuation_packet(packet)

    existing_ids = {step["id"] for step in _sorted_plan_steps(plan)}
    normalized_steps: list[dict] = []
    for idx, raw_step in enumerate(raw_steps, start=1):
        step = _coerce_planned_step(raw_step, fallback_prefix=f"{id_prefix}_{idx}", index=idx)
        _ensure_unique_step_ids(existing_ids, step, prefix=f"{id_prefix}_{idx}")
        normalized_steps.append(step)

    if normalized_steps:
        trial_plan = {
            "plan_id": plan.get("plan_id", "plan"),
            "session_name": plan.get("session_name", "session"),
            "steps": _sorted_plan_steps(plan) + normalized_steps,
            "final_output": plan.get("final_output", "final.json"),
        }
        validate_plan(trial_plan, for_execution=True)
    return normalized_steps, reason


def _full_replan_prompt(goal: str, state_summary: dict, failure_reason: str) -> str:
    payload = {
        "task": "Create a full replacement plan to complete the original goal.",
        "requirements": [
            "Return JSON object for a full plan.",
            "Plan must remain aligned with original goal.",
            "Use minimal steps and deterministic tool arguments.",
        ],
        "goal": goal,
        "failure_reason": failure_reason,
        "state_summary": state_summary,
    }
    return json.dumps(payload, ensure_ascii=False)


def _generate_full_replan(
    *,
    goal: str,
    state: dict,
    failure_reason: str,
    model_path: str,
    n_ctx: int,
    temp: float,
    seed: int,
    log_dir: Path,
) -> dict:
    replan_prompt = _full_replan_prompt(goal, _state_summary_for_planning(state), failure_reason)
    plan = generate_plan(
        replan_prompt,
        model_path=model_path,
        n_ctx=n_ctx,
        temp=temp,
        seed=seed,
        llm_log_dir=log_dir,
    )
    validate_plan(plan, for_execution=False)
    return plan


def _run_agent_loop_legacy(
    *,
    session_dir: Path,
    original_prompt: str,
    initial_plan: dict,
    meta: dict,
    model_path: str,
    n_ctx: int,
    temp: float,
    seed: int,
    max_steps: int,
) -> dict:
    session_dir.mkdir(parents=True, exist_ok=True)
    iterations_dir = session_dir / "iterations"
    iterations_dir.mkdir(parents=True, exist_ok=True)

    session_log: dict = {
        "mode": "one_shot",
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
            raw_analysis, analysis, analysis_note = _generate_analysis(
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
            raw_analysis = None
            analysis_note = f"analysis_failed: {exc}"
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

        write_json(
            iter_dir / "analysis_raw.json",
            {
                "raw_analysis": raw_analysis,
                "note": analysis_note,
            },
        )
        write_json(iter_dir / "analysis_norm.json", analysis)
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
        "mode": "one_shot",
    }

def _run_agent_loop_multi_step(
    *,
    session_dir: Path,
    original_prompt: str,
    initial_plan: dict,
    meta: dict,
    model_path: str,
    n_ctx: int,
    temp: float,
    seed: int,
    max_steps: int,
    max_retries: int,
    max_expansions: int,
    max_replans: int,
    max_runtime_sec: int | None,
) -> dict:
    session_dir.mkdir(parents=True, exist_ok=True)
    validate_plan(initial_plan, for_execution=False)

    started_at = datetime.now(timezone.utc)
    history_dir = session_dir / "multi_step"
    history_dir.mkdir(parents=True, exist_ok=True)

    state: dict = {
        "mode": "multi_step",
        "planning_mode": "initial",
        "goal": original_prompt,
        "current_plan": initial_plan,
        "completed_steps": [],
        "pending_steps": [step["id"] for step in _sorted_plan_steps(initial_plan)],
        "failed_steps": [],
        "tool_outputs": {},
        "artifacts": {},
        "step_attempts": {},
        "continuation_expansions": 0,
        "replanning_attempts": 0,
        "run_count": 0,
        "limits": {
            "max_steps": max_steps,
            "max_retries": max_retries,
            "max_expansions": max_expansions,
            "max_replans": max_replans,
            "max_runtime_sec": max_runtime_sec,
        },
        "log": {
            "original_plan_steps": _sorted_plan_steps(initial_plan),
            "appended_steps": [],
            "replans": [],
            "step_runs": [],
        },
        "summary": "",
        "started_at": _utc_now(),
    }
    write_json(session_dir / "session_state.json", state)

    while True:
        elapsed_sec = int((datetime.now(timezone.utc) - started_at).total_seconds())
        if max_runtime_sec is not None and elapsed_sec >= max_runtime_sec:
            state["summary"] = "stopped: runtime limit reached"
            break
        if state["run_count"] >= max_steps:
            state["summary"] = "stopped: max_steps reached"
            break

        completed_set = set(state["completed_steps"])
        failed_set = {item["step_id"] for item in state["failed_steps"] if isinstance(item, dict)}
        skipped_due_to_failed_dependency: set[str] = set()
        for step in _sorted_plan_steps(state["current_plan"]):
            step_id = step["id"]
            if step_id in completed_set or step_id in failed_set:
                continue
            deps = step.get("depends_on", [])
            if any(dep in failed_set for dep in deps):
                skipped_due_to_failed_dependency.add(step_id)

        pending = [
            step["id"]
            for step in _sorted_plan_steps(state["current_plan"])
            if step["id"] not in completed_set and step["id"] not in failed_set and step["id"] not in skipped_due_to_failed_dependency
        ]
        state["pending_steps"] = pending

        step = _find_next_executable_step(
            state["current_plan"],
            completed=completed_set,
            failed=failed_set,
            queued=set(),
        )

        if step is None:
            if pending:
                if state["replanning_attempts"] >= max_replans:
                    state["summary"] = "stopped: invalid dependency graph and replan limit reached"
                    break
                state["planning_mode"] = "full_replan"
                replanned = _generate_full_replan(
                    goal=original_prompt,
                    state=state,
                    failure_reason="No executable step due to dependencies",
                    model_path=model_path,
                    n_ctx=n_ctx,
                    temp=temp,
                    seed=seed,
                    log_dir=history_dir / f"replan_{state['replanning_attempts'] + 1:03d}",
                )
                state["current_plan"] = replanned
                state["replanning_attempts"] += 1
                state["log"]["replans"].append(
                    {
                        "index": state["replanning_attempts"],
                        "reason": "No executable step due to dependencies",
                        "plan": replanned,
                    }
                )
                write_json(session_dir / "session_state.json", state)
                continue

            if state["continuation_expansions"] < max_expansions:
                state["planning_mode"] = "continuation_append"
                new_steps, reason = _generate_continuation_steps(
                    goal=original_prompt,
                    plan=state["current_plan"],
                    state=state,
                    model_path=model_path,
                    n_ctx=n_ctx,
                    temp=temp,
                    seed=seed,
                    log_dir=history_dir / f"continuation_{state['continuation_expansions'] + 1:03d}",
                    id_prefix=f"cont_{state['continuation_expansions'] + 1:03d}",
                )
                state["continuation_expansions"] += 1
                if new_steps:
                    state["current_plan"]["steps"].extend(new_steps)
                    state["log"]["appended_steps"].append(
                        {"reason": reason, "steps": new_steps, "at_run": state["run_count"]}
                    )
                    validate_plan(state["current_plan"], for_execution=False)
                    write_json(session_dir / "session_state.json", state)
                    continue
            state["summary"] = "success"
            break

        step_id = step["id"]
        attempt = int(state["step_attempts"].get(step_id, 0)) + 1
        state["step_attempts"][step_id] = attempt
        state["run_count"] += 1

        run_dir = history_dir / f"run_{state['run_count']:03d}_{step_id}_try{attempt}"
        run_dir.mkdir(parents=True, exist_ok=True)
        step_plan = _step_plan(state["current_plan"], step, state["run_count"], attempt)
        validate_plan(step_plan, for_execution=True)
        write_plan_and_meta(run_dir, step_plan, meta)
        execution = execute_plan(step_plan, run_dir)

        step_result = execution["results"][0] if execution.get("results") else {
            "id": step_id,
            "tool": step.get("tool"),
            "exit_code": -1,
            "timed_out": False,
            "elapsed_sec": 0.0,
            "stdout_file": "",
            "stderr_file": "",
        }

        valid, failures, io = _validate_step_result(step, step_result)
        state["tool_outputs"][step_id] = {
            "exit_code": step_result.get("exit_code"),
            "timed_out": step_result.get("timed_out"),
            "elapsed_sec": step_result.get("elapsed_sec"),
            "stdout_excerpt": io["stdout"],
            "stderr_excerpt": io["stderr"],
            "stdout_truncated": io["stdout_truncated"],
            "stderr_truncated": io["stderr_truncated"],
        }
        state["artifacts"][step_id] = [
            step_result.get("stdout_file"),
            step_result.get("stderr_file"),
        ]

        state["log"]["step_runs"].append(
            {
                "run_index": state["run_count"],
                "step_id": step_id,
                "description": step.get("description"),
                "planning_mode": state["planning_mode"],
                "attempt": attempt,
                "success": valid,
                "validation_failures": failures,
                "result": step_result,
                "artifacts": state["artifacts"][step_id],
            }
        )

        if valid:
            if step_id not in state["completed_steps"]:
                state["completed_steps"].append(step_id)
            state["planning_mode"] = "continuation_append"
            write_json(session_dir / "session_state.json", state)
            continue

        allowed_retries = min(max_retries, int(step.get("retry_policy", {}).get("max_retries", 0)))
        if attempt <= allowed_retries:
            state["planning_mode"] = "initial"
            write_json(session_dir / "session_state.json", state)
            continue

        state["failed_steps"].append(
            {
                "step_id": step_id,
                "attempt": attempt,
                "validation_failures": failures,
                "result": step_result,
            }
        )

        continuation_added = False
        if state["continuation_expansions"] < max_expansions:
            state["planning_mode"] = "continuation_append"
            new_steps, reason = _generate_continuation_steps(
                goal=original_prompt,
                plan=state["current_plan"],
                state=state,
                model_path=model_path,
                n_ctx=n_ctx,
                temp=temp,
                seed=seed,
                log_dir=history_dir / f"continuation_{state['continuation_expansions'] + 1:03d}",
                id_prefix=f"recover_{state['continuation_expansions'] + 1:03d}",
            )
            state["continuation_expansions"] += 1
            if new_steps:
                continuation_added = True
                state["current_plan"]["steps"].extend(new_steps)
                state["log"]["appended_steps"].append(
                    {"reason": reason, "steps": new_steps, "at_run": state["run_count"]}
                )
                validate_plan(state["current_plan"], for_execution=False)

        if continuation_added:
            write_json(session_dir / "session_state.json", state)
            continue

        if state["replanning_attempts"] < max_replans:
            state["planning_mode"] = "full_replan"
            replanned = _generate_full_replan(
                goal=original_prompt,
                state=state,
                failure_reason=f"Step failed after retries: {step_id}",
                model_path=model_path,
                n_ctx=n_ctx,
                temp=temp,
                seed=seed,
                log_dir=history_dir / f"replan_{state['replanning_attempts'] + 1:03d}",
            )
            state["replanning_attempts"] += 1
            state["current_plan"] = replanned
            state["log"]["replans"].append(
                {
                    "index": state["replanning_attempts"],
                    "reason": f"Step failed after retries: {step_id}",
                    "plan": replanned,
                }
            )
            write_json(session_dir / "session_state.json", state)
            continue

        state["summary"] = f"stopped: step {step_id} failed and limits reached"
        break

    state["finished_at"] = _utc_now()
    write_json(session_dir / "session_state.json", state)

    session_log = {
        "mode": "multi_step",
        "summary": state["summary"],
        "goal": original_prompt,
        "completed_steps": state["completed_steps"],
        "failed_steps": state["failed_steps"],
        "pending_steps": state["pending_steps"],
        "continuation_expansions": state["continuation_expansions"],
        "replanning_attempts": state["replanning_attempts"],
        "run_count": state["run_count"],
        "log": state["log"],
        "session_dir": str(session_dir),
        "started_at": state["started_at"],
        "finished_at": state["finished_at"],
    }
    write_json(session_dir / "session.json", session_log)
    write_json(session_dir / "final.json", session_log)
    return {
        "summary": state["summary"],
        "iterations": state["run_count"],
        "session_dir": str(session_dir),
        "mode": "multi_step",
    }


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
    multi_step_mode: bool = False,
    max_retries: int = 1,
    max_expansions: int = 2,
    max_replans: int = 1,
    max_runtime_sec: int | None = None,
) -> dict:
    if not multi_step_mode:
        return _run_agent_loop_legacy(
            session_dir=session_dir,
            original_prompt=original_prompt,
            initial_plan=initial_plan,
            meta=meta,
            model_path=model_path,
            n_ctx=n_ctx,
            temp=temp,
            seed=seed,
            max_steps=max_steps,
        )
    return _run_agent_loop_multi_step(
        session_dir=session_dir,
        original_prompt=original_prompt,
        initial_plan=initial_plan,
        meta=meta,
        model_path=model_path,
        n_ctx=n_ctx,
        temp=temp,
        seed=seed,
        max_steps=max_steps,
        max_retries=max_retries,
        max_expansions=max_expansions,
        max_replans=max_replans,
        max_runtime_sec=max_runtime_sec,
    )
