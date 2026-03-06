from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.orchestrator.llm import LLMError, extract_json_object, run_llm_text
from src.orchestrator.tools import REGISTRY, list_planning_tools, validate_tool_args

SCHEMA_PATH = Path(__file__).with_name("plan.schema.json")


class PlanValidationError(ValueError):
    pass


DEFAULT_SUCCESS_CRITERIA = {"type": "exit_code", "equals": 0}
DEFAULT_RETRY_POLICY = {"max_retries": 0}


def sanitize_session_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", (value or "session").strip()).strip("._-")
    if not cleaned:
        cleaned = "session"
    return cleaned[:80]


def load_plan_schema() -> dict:
    with SCHEMA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _expect_type(name: str, value, typ) -> None:
    if not isinstance(value, typ):
        raise PlanValidationError(f"{name} must be {typ.__name__}")


def _strip_legacy_outputs(plan: dict) -> None:
    if not isinstance(plan, dict):
        return
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return
    for step in steps:
        if isinstance(step, dict):
            step.pop("outputs", None)


def _validate_against_schema(plan: dict) -> None:
    if not isinstance(plan, dict):
        raise PlanValidationError("schema validation failed: plan must be object")

    allowed_top = {"plan_id", "session_name", "steps", "final_output", "notes"}
    extra_top = set(plan.keys()) - allowed_top
    if extra_top:
        keys = ", ".join(sorted(extra_top))
        raise PlanValidationError(f"schema validation failed: unknown top-level fields: {keys}")

    steps = plan.get("steps")
    if steps is None:
        return
    if not isinstance(steps, list):
        raise PlanValidationError("schema validation failed: steps must be array")

    allowed_step = {
        "id",
        "description",
        "tool",
        "args",
        "timeout_sec",
        "depends_on",
        "success_criteria",
        "retry_policy",
    }
    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise PlanValidationError(f"schema validation failed: steps[{idx}] must be object")
        extra_step = set(step.keys()) - allowed_step
        if extra_step:
            keys = ", ".join(sorted(extra_step))
            raise PlanValidationError(
                f"schema validation failed: steps[{idx}] unknown fields: {keys}"
            )


def _normalize_step_defaults(plan: dict) -> None:
    if not isinstance(plan, dict):
        return
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return

    for step in steps:
        if not isinstance(step, dict):
            continue

        tool_name = step.get("tool")
        step_id = step.get("id")
        if not isinstance(step.get("description"), str) or not step.get("description", "").strip():
            if isinstance(tool_name, str) and tool_name.strip():
                step["description"] = f"Execute tool '{tool_name}'"
            elif isinstance(step_id, str) and step_id.strip():
                step["description"] = f"Execute step '{step_id}'"
            else:
                step["description"] = "Execute step"

        depends_on = step.get("depends_on")
        if depends_on is None:
            step["depends_on"] = []

        success_criteria = step.get("success_criteria")
        if not isinstance(success_criteria, dict):
            step["success_criteria"] = dict(DEFAULT_SUCCESS_CRITERIA)
        else:
            merged_success = dict(DEFAULT_SUCCESS_CRITERIA)
            merged_success.update(success_criteria)
            step["success_criteria"] = merged_success

        retry_policy = step.get("retry_policy")
        if not isinstance(retry_policy, dict):
            step["retry_policy"] = dict(DEFAULT_RETRY_POLICY)
        else:
            merged_retry = dict(DEFAULT_RETRY_POLICY)
            merged_retry.update(retry_policy)
            step["retry_policy"] = merged_retry


def _validate_success_criteria(name: str, payload: dict) -> None:
    if not isinstance(payload, dict):
        raise PlanValidationError(f"{name} must be object")
    criteria_type = payload.get("type")
    if criteria_type != "exit_code":
        raise PlanValidationError(f"{name}.type must be 'exit_code'")
    equals = payload.get("equals")
    if not isinstance(equals, int):
        raise PlanValidationError(f"{name}.equals must be integer")
    for field in ("stdout_contains", "stderr_contains"):
        value = payload.get(field, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise PlanValidationError(f"{name}.{field} must be string[]")


def _validate_retry_policy(name: str, payload: dict) -> None:
    if not isinstance(payload, dict):
        raise PlanValidationError(f"{name} must be object")
    max_retries = payload.get("max_retries")
    if not isinstance(max_retries, int) or max_retries < 0 or max_retries > 10:
        raise PlanValidationError(f"{name}.max_retries must be integer 0..10")


def validate_plan(plan: dict, *, for_execution: bool) -> None:
    _strip_legacy_outputs(plan)
    _normalize_step_defaults(plan)
    _validate_against_schema(plan)
    _expect_type("plan", plan, dict)
    for required in ("plan_id", "session_name", "steps", "final_output"):
        if required not in plan:
            raise PlanValidationError(f"missing field: {required}")

    _expect_type("plan_id", plan["plan_id"], str)
    _expect_type("session_name", plan["session_name"], str)
    _expect_type("steps", plan["steps"], list)
    _expect_type("final_output", plan["final_output"], str)

    if not plan["plan_id"].strip():
        raise PlanValidationError("plan_id cannot be empty")
    if not plan["final_output"].strip():
        raise PlanValidationError("final_output cannot be empty")

    plan["session_name"] = sanitize_session_name(plan["session_name"])

    seen_ids: set[str] = set()
    for idx, step in enumerate(plan["steps"], start=1):
        _expect_type(f"steps[{idx}]", step, dict)
        for field in ("id", "description", "tool", "args", "success_criteria", "retry_policy"):
            if field not in step:
                raise PlanValidationError(f"steps[{idx}] missing field: {field}")
        _expect_type(f"steps[{idx}].id", step["id"], str)
        _expect_type(f"steps[{idx}].description", step["description"], str)
        _expect_type(f"steps[{idx}].tool", step["tool"], str)
        _expect_type(f"steps[{idx}].args", step["args"], dict)
        _validate_success_criteria(f"steps[{idx}].success_criteria", step["success_criteria"])
        _validate_retry_policy(f"steps[{idx}].retry_policy", step["retry_policy"])

        step_id = step["id"].strip()
        if not step_id:
            raise PlanValidationError(f"steps[{idx}].id cannot be empty")
        if step_id in seen_ids:
            raise PlanValidationError(f"duplicate step id: {step_id}")
        seen_ids.add(step_id)

        tool_name = step["tool"]
        tool = REGISTRY.get(tool_name)
        if tool is None:
            raise PlanValidationError(f"steps[{idx}] unknown tool: {tool_name}")
        if for_execution and not tool.enabled_for_execution:
            raise PlanValidationError(f"steps[{idx}] tool disabled for execution: {tool_name}")
        if not for_execution and not tool.enabled_for_planning:
            raise PlanValidationError(f"steps[{idx}] tool disabled for planning: {tool_name}")

        validate_tool_args(tool_name, step["args"])

        timeout = step.get("timeout_sec")
        if timeout is not None and (not isinstance(timeout, int) or timeout < 1 or timeout > 3600):
            raise PlanValidationError(f"steps[{idx}].timeout_sec must be integer 1..3600")

        depends_on = step.get("depends_on", [])
        if not isinstance(depends_on, list) or any(
            not isinstance(item, str) or not item.strip() for item in depends_on
        ):
            raise PlanValidationError(f"steps[{idx}].depends_on must be non-empty string[]")

    known_ids = {step["id"] for step in plan["steps"] if isinstance(step, dict) and "id" in step}
    for idx, step in enumerate(plan["steps"], start=1):
        depends_on = step.get("depends_on", [])
        unknown = [dep for dep in depends_on if dep not in known_ids]
        if unknown:
            raise PlanValidationError(
                f"steps[{idx}].depends_on references unknown steps: {', '.join(unknown)}"
            )

    notes = plan.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise PlanValidationError("notes must be string")


def fallback_plan(user_prompt: str, reason: str) -> dict:
    prompt_snippet = (user_prompt or "").strip().replace("\n", " ")[:120]
    return {
        "plan_id": f"fallback_{uuid.uuid4().hex[:12]}",
        "session_name": sanitize_session_name(prompt_snippet or "fallback"),
        "steps": [],
        "final_output": "final.json",
        "notes": f"planner_failed: {reason}",
    }


def _planner_prompt(user_prompt: str, tools_meta: list[dict]) -> str:
    schema = load_plan_schema()
    envelope = {
        "task": "Generate a deterministic tool execution plan.",
        "requirements": [
            "Return only one JSON object. No markdown. No prose.",
            "Do not use markdown code fences or ```json blocks.",
            "Do not add explanations before or after JSON.",
            "If unable to comply, return exactly: {\"error\":\"cannot_comply\"}",
            "Only use tools listed in tools_meta.",
            "Use steps only when a tool is required.",
            "Prefer minimal step count.",
            "Each step must include a natural-language description.",
            "Each step must include success_criteria with type=exit_code and equals=0 unless caller requires stricter checks.",
            "Use depends_on only when a step requires prior step outputs.",
            "Set retry_policy.max_retries conservatively (usually 0 or 1).",
            "Plan includes only tool choices and arguments (and optional timeout_sec); never include execution results.",
            "Do not output latency, reachability, success/failure, or any measured runtime values.",
            "Do not include outputs, file paths, or artifact locations in steps.",
            "Set final_output to final.json unless caller requires a different extension.",
        ],
        "utc_now": datetime.now(timezone.utc).isoformat(),
        "user_prompt": user_prompt,
        "tools_meta": tools_meta,
        "target_schema": schema,
    }
    return json.dumps(envelope, ensure_ascii=False)


def generate_plan(
    user_prompt: str,
    *,
    model_path: str,
    n_ctx: int = 2048,
    temp: float = 0.0,
    seed: int = 0,
    llm_log_dir: str | Path | None = None,
) -> dict:
    tools_meta = list_planning_tools()
    prompt = _planner_prompt(user_prompt=user_prompt, tools_meta=tools_meta)
    try:
        text = run_llm_text(
            prompt,
            SCHEMA_PATH,
            model_path=model_path,
            n_ctx=n_ctx,
            temp=temp,
            seed=seed,
            log_dir=llm_log_dir,
        )
        plan = extract_json_object(text)
        if not isinstance(plan, dict):
            raise LLMError("model returned non-object")
        if plan.get("error") == "cannot_comply":
            raise LLMError("model returned cannot_comply")
        validate_plan(plan, for_execution=False)
        if not plan.get("plan_id"):
            plan["plan_id"] = f"plan_{uuid.uuid4().hex[:12]}"
        plan["session_name"] = sanitize_session_name(plan.get("session_name", "session"))
        if not plan.get("final_output"):
            plan["final_output"] = "final.json"
        return plan
    except Exception as exc:
        return fallback_plan(user_prompt=user_prompt, reason=str(exc))
