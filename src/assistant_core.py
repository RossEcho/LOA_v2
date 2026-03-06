from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator.llm import extract_json_object, run_llm_text

DECISION_SCHEMA_PATH = PROJECT_ROOT / "protocol" / "assistant_decision.schema.json"
FINAL_SCHEMA_PATH = PROJECT_ROOT / "protocol" / "assistant_response.schema.json"
DEFAULT_BRIDGE_PATH = PROJECT_ROOT / "bin" / "loa-bridge"
ACTION_CLASSES = {"READ", "WRITE", "NETWORK", "SYSTEM"}


class AssistantError(RuntimeError):
    pass


def _default_bridge_json(args: list[str], payload: dict | None = None) -> dict | list:
    cmd = [sys.executable, str(DEFAULT_BRIDGE_PATH)] + args
    input_text = None if payload is None else json.dumps(payload, ensure_ascii=False)
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )
    raw = (proc.stdout or "").strip()
    if not raw:
        raise AssistantError(f"bridge returned empty output: {proc.stderr.strip()}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssistantError(f"bridge returned invalid JSON: {raw}") from exc


class AssistantCore:
    def __init__(
        self,
        *,
        model_path: str | None = None,
        n_ctx: int = 2048,
        temp: float = 0.0,
        seed: int = 0,
        bridge_json_runner: Callable[[list[str], dict | None], dict | list] | None = None,
        llm_text_runner: Callable[..., str] | None = None,
    ) -> None:
        self.model_path = model_path or os.getenv("LOA_MODEL_PATH", "local")
        self.n_ctx = n_ctx
        self.temp = temp
        self.seed = seed
        self._bridge_json = bridge_json_runner or _default_bridge_json
        self._llm_text = llm_text_runner or run_llm_text
        self.tools = self._load_tools()

    def _load_tools(self) -> list[dict]:
        payload = self._bridge_json(["--list-tools"], None)
        if not isinstance(payload, list):
            raise AssistantError("bridge --list-tools must return a JSON list")
        for item in payload:
            if not isinstance(item, dict):
                raise AssistantError("tool entry must be an object")
            if item.get("action_class") not in ACTION_CLASSES:
                raise AssistantError(f"tool {item.get('name')} has invalid action_class")
        return payload

    def _prompt_for_decision(self, user_input: str) -> str:
        envelope = {
            "task": "Choose whether to reply directly or call one tool.",
            "requirements": [
                "Return one JSON object only.",
                "If a tool is required, set action='tool' and provide tool_name, args, action_class.",
                "If no tool is required, set action='respond' and provide response.",
                "Never invent tool names. Use only tools listed below.",
            ],
            "tools": self.tools,
            "user_input": user_input,
            "output_schema": {
                "action": "respond|tool",
                "response": "string|null",
                "tool_name": "string|null",
                "args": "object|null",
                "action_class": "READ|WRITE|NETWORK|SYSTEM|null",
                "timeout_seconds": "number|null",
            },
        }
        return json.dumps(envelope, ensure_ascii=False)

    def _prompt_for_final_response(self, user_input: str, tool_call: dict, tool_result: dict) -> str:
        envelope = {
            "task": "Write a concise user-facing response.",
            "requirements": [
                "Return one JSON object only with field: response.",
                "If tool failed, explain the failure clearly.",
                "If tool succeeded, summarize relevant output.",
            ],
            "user_input": user_input,
            "tool_call": tool_call,
            "tool_result": tool_result,
        }
        return json.dumps(envelope, ensure_ascii=False)

    def _decide(self, user_input: str) -> dict:
        text = self._llm_text(
            self._prompt_for_decision(user_input),
            DECISION_SCHEMA_PATH,
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            temp=self.temp,
            seed=self.seed,
            log_dir=None,
        )
        decision = extract_json_object(text)
        if not isinstance(decision, dict):
            raise AssistantError("decision must be a JSON object")
        return decision

    def _validate_decision(self, decision: dict) -> None:
        action = decision.get("action")
        if action not in {"respond", "tool"}:
            raise AssistantError("decision.action must be 'respond' or 'tool'")
        if action == "respond":
            if not isinstance(decision.get("response"), str):
                raise AssistantError("decision.response must be a string for action=respond")
            return
        if not isinstance(decision.get("tool_name"), str) or not decision["tool_name"].strip():
            raise AssistantError("decision.tool_name must be a non-empty string for action=tool")
        args = decision.get("args")
        if not isinstance(args, dict):
            raise AssistantError("decision.args must be an object for action=tool")
        action_class = decision.get("action_class")
        if action_class not in ACTION_CLASSES:
            raise AssistantError("decision.action_class invalid for action=tool")
        known_tools = {tool["name"] for tool in self.tools}
        if decision["tool_name"] not in known_tools:
            raise AssistantError(f"unknown tool in decision: {decision['tool_name']}")

    def _is_ping_request(self, user_input: str) -> bool:
        return isinstance(user_input, str) and "ping" in user_input.lower()

    def _extract_ping_targets(self, user_input: str, max_targets: int = 16) -> tuple[list[str], bool]:
        if not isinstance(user_input, str):
            return [], False

        lowered = user_input.lower()
        targets: list[str] = []
        seen: set[str] = set()
        truncated = False

        range_pattern = re.compile(r"\b((?:\d{1,3}\.){3})(\d{1,3})\s*-\s*(\d{1,3})\b")
        for match in range_pattern.finditer(lowered):
            prefix = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))
            lo = max(1, min(start, end))
            hi = min(254, max(start, end))
            for octet in range(lo, hi + 1):
                target = f"{prefix}{octet}"
                if target in seen:
                    continue
                targets.append(target)
                seen.add(target)
                if len(targets) >= max_targets:
                    truncated = True
                    return targets, truncated

        ip_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
        host_pattern = re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b")

        for pattern in (ip_pattern, host_pattern):
            for match in pattern.finditer(user_input):
                value = match.group(0)
                if value in seen:
                    continue
                targets.append(value)
                seen.add(value)
                if len(targets) >= max_targets:
                    truncated = True
                    return targets, truncated

        return targets, truncated

    def _ping_decisions_from_input(self, user_input: str, base_decision: dict | None = None) -> tuple[list[dict], bool]:
        targets, truncated = self._extract_ping_targets(user_input)
        if not targets and isinstance(base_decision, dict):
            args = base_decision.get("args")
            target = args.get("target") if isinstance(args, dict) else None
            if isinstance(target, str) and target.strip():
                targets = [target.strip()]

        if not targets:
            return [], truncated

        timeout = None
        if isinstance(base_decision, dict):
            timeout = base_decision.get("timeout_seconds")

        decisions: list[dict] = []
        for target in targets:
            args = {"target": target, "count": 1}
            decisions.append(
                {
                    "action": "tool",
                    "tool_name": "ping",
                    "args": args,
                    "action_class": "NETWORK",
                    "timeout_seconds": timeout,
                    "response": None,
                }
            )
        return decisions, truncated

    def _run_tool(self, decision: dict) -> dict:
        call = {
            "tool_name": decision["tool_name"],
            "args": decision["args"],
            "cwd": str(PROJECT_ROOT),
            "timeout_seconds": decision.get("timeout_seconds"),
            "action_class": decision["action_class"],
            "env": None,
        }
        result = self._bridge_json([], call)
        if not isinstance(result, dict):
            raise AssistantError("bridge call result must be a JSON object")
        return {"call": call, "result": result}

    def _summarize_batch_ping(self, tool_execs: list[dict], *, truncated: bool) -> str:
        ok = 0
        failed = 0
        details: list[str] = []
        for item in tool_execs:
            call = item["call"]
            result = item["result"]
            target = call["args"].get("target", "unknown")
            if result.get("ok"):
                ok += 1
                details.append(f"{target}: ok")
            else:
                failed += 1
                err = (result.get("stderr") or "").strip()
                details.append(f"{target}: failed ({err or 'no error details'})")
        prefix = f"Ping summary: {ok} succeeded, {failed} failed."
        if truncated:
            prefix += " Target list was truncated to keep execution bounded."
        return prefix + "\n" + "\n".join(details)

    def _finalize_response(self, user_input: str, tool_call: dict, tool_result: dict) -> str:
        text = self._llm_text(
            self._prompt_for_final_response(user_input, tool_call, tool_result),
            FINAL_SCHEMA_PATH,
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            temp=self.temp,
            seed=self.seed,
            log_dir=None,
        )
        payload = extract_json_object(text)
        response = payload.get("response")
        if not isinstance(response, str):
            raise AssistantError("final response payload missing response")
        return response

    def handle_user_input(self, user_input: str) -> dict:
        decision = self._decide(user_input)
        self._validate_decision(decision)

        if self._is_ping_request(user_input):
            ping_decisions: list[dict] = []
            truncated = False
            if decision.get("action") == "respond":
                ping_decisions, truncated = self._ping_decisions_from_input(user_input)
            elif decision.get("action") == "tool" and decision.get("tool_name") == "ping":
                ping_decisions, truncated = self._ping_decisions_from_input(user_input, decision)

            if ping_decisions:
                if len(ping_decisions) == 1:
                    tool_exec = self._run_tool(ping_decisions[0])
                    response = self._finalize_response(user_input, tool_exec["call"], tool_exec["result"])
                    logs = [f"decision override: tool=ping targets=1"]
                    preview = tool_exec["result"].get("command_preview")
                    if isinstance(preview, str) and preview.strip():
                        logs.append(f"command: {preview}")
                    return {
                        "response": response,
                        "decision": ping_decisions[0],
                        "tool_call": tool_exec["call"],
                        "tool_result": tool_exec["result"],
                        "logs": logs,
                    }

                tool_execs = [self._run_tool(ping_decision) for ping_decision in ping_decisions]
                logs = [f"decision override: tool=ping targets={len(tool_execs)}"]
                for item in tool_execs:
                    preview = item["result"].get("command_preview")
                    if isinstance(preview, str) and preview.strip():
                        logs.append(f"command: {preview}")
                return {
                    "response": self._summarize_batch_ping(tool_execs, truncated=truncated),
                    "decision": {"action": "tool_batch", "tool_name": "ping", "count": len(tool_execs)},
                    "tool_call": [item["call"] for item in tool_execs],
                    "tool_result": [item["result"] for item in tool_execs],
                    "logs": logs,
                }

        if decision["action"] == "respond":
            return {
                "response": decision["response"],
                "decision": decision,
                "tool_call": None,
                "tool_result": None,
                "logs": [f"decision: respond"],
            }

        tool_exec = self._run_tool(decision)
        response = self._finalize_response(user_input, tool_exec["call"], tool_exec["result"])
        logs = [f"decision: tool={decision['tool_name']} action_class={decision['action_class']}"]
        preview = tool_exec["result"].get("command_preview")
        if isinstance(preview, str) and preview.strip():
            logs.append(f"command: {preview}")
        return {
            "response": response,
            "decision": decision,
            "tool_call": tool_exec["call"],
            "tool_result": tool_exec["result"],
            "logs": logs,
        }
