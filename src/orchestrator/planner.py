from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.orchestrator.llm import LLMError, run_llm_json
from src.orchestrator.tools import REGISTRY, list_planning_tools, validate_tool_args

SCHEMA_PATH = Path(__file__).with_name("plan.schema.json")


class PlanValidationError(ValueError):
    pass


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

    allowed_step = {"id", "tool", "args", "timeout_sec", "outputs"}
    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise PlanValidationError(f"schema validation failed: steps[{idx}] must be object")
        extra_step = set(step.keys()) - allowed_step
        if extra_step:
            keys = ", ".join(sorted(extra_step))
            raise PlanValidationError(
                f"schema validation failed: steps[{idx}] unknown fields: {keys}"
            )


def validate_plan(plan: dict, *, for_execution: bool) -> None:
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
        for field in ("id", "tool", "args"):
            if field not in step:
                raise PlanValidationError(f"steps[{idx}] missing field: {field}")
        _expect_type(f"steps[{idx}].id", step["id"], str)
        _expect_type(f"steps[{idx}].tool", step["tool"], str)
        _expect_type(f"steps[{idx}].args", step["args"], dict)

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

        outputs = step.get("outputs")
        if outputs is not None:
            _expect_type(f"steps[{idx}].outputs", outputs, dict)
            for key, value in outputs.items():
                if not isinstance(key, str) or not key:
                    raise PlanValidationError(f"steps[{idx}].outputs contains invalid key")
                if not isinstance(value, str) or not value.strip():
                    raise PlanValidationError(f"steps[{idx}].outputs[{key}] must be non-empty string")

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
            "Only use tools listed in tools_meta.",
            "Use steps only when a tool is required.",
            "Prefer minimal step count.",
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
) -> dict:
    tools_meta = list_planning_tools()
    prompt = _planner_prompt(user_prompt=user_prompt, tools_meta=tools_meta)
    try:
        plan = run_llm_json(
            prompt,
            SCHEMA_PATH,
            model_path=model_path,
            n_ctx=n_ctx,
            temp=temp,
            seed=seed,
        )
        if not isinstance(plan, dict):
            raise LLMError("model returned non-object")
        validate_plan(plan, for_execution=False)
        if not plan.get("plan_id"):
            plan["plan_id"] = f"plan_{uuid.uuid4().hex[:12]}"
        plan["session_name"] = sanitize_session_name(plan.get("session_name", "session"))
        if not plan.get("final_output"):
            plan["final_output"] = "final.json"
        return plan
    except Exception as exc:
        return fallback_plan(user_prompt=user_prompt, reason=str(exc))
