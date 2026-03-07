import json
import unittest
from unittest.mock import patch

from src.assistant_core import AssistantCore


class TestAssistantCore(unittest.TestCase):
    def test_tool_loop(self):
        calls = {"llm": 0}

        def fake_bridge(args, payload):
            if args == ["--list-tools"]:
                return [
                    {
                        "name": "ping",
                        "version": "1.0.0",
                        "description": "desc",
                        "action_class": "NETWORK",
                        "args_schema": {"type": "object"},
                    }
                ]
            return {
                "ok": True,
                "exit_code": 0,
                "stdout": "pong",
                "stderr": "",
                "duration_ms": 10,
                "artifacts": [],
            }

        def fake_llm(prompt, schema_path, **kwargs):
            calls["llm"] += 1
            if calls["llm"] == 1:
                return json.dumps(
                    {
                        "action": "tool",
                        "tool_name": "ping",
                        "args": {"target": "8.8.8.8"},
                        "action_class": "NETWORK",
                        "timeout_seconds": 3,
                        "response": None,
                    }
                )
            return json.dumps({"action": "respond", "response": "Ping completed.", "tool_name": None, "args": None, "action_class": None, "timeout_seconds": None})

        assistant = AssistantCore(bridge_json_runner=fake_bridge, llm_text_runner=fake_llm)
        result = assistant.handle_user_input("ping 8.8.8.8")
        self.assertEqual(result["response"], "Ping completed.")
        self.assertEqual(len(result["tool_call"]), 1)
        self.assertEqual(result["tool_call"][0]["tool_name"], "ping")
        self.assertTrue(result["tool_result"][0]["ok"])

    def test_direct_response(self):
        def fake_bridge(args, payload):
            return [
                {
                    "name": "ping",
                    "version": "1.0.0",
                    "description": "desc",
                    "action_class": "NETWORK",
                    "args_schema": {"type": "object"},
                }
            ]

        def fake_llm(prompt, schema_path, **kwargs):
            return json.dumps({"action": "respond", "response": "Hello", "tool_name": None, "args": None, "action_class": None, "timeout_seconds": None})

        assistant = AssistantCore(bridge_json_runner=fake_bridge, llm_text_runner=fake_llm)
        result = assistant.handle_user_input("hi")
        self.assertEqual(result["response"], "Hello")
        self.assertIsNone(result["tool_call"])

    def test_ping_range_runs_batch_after_tool_decision(self):
        calls: list[dict] = []
        llm_calls = {"count": 0}

        def fake_bridge(args, payload):
            if args == ["--list-tools"]:
                return [
                    {
                        "name": "ping",
                        "version": "1.0.0",
                        "description": "desc",
                        "action_class": "NETWORK",
                        "args_schema": {"type": "object"},
                    }
                ]
            calls.append(payload)
            return {
                "ok": True,
                "exit_code": 0,
                "stdout": "pong",
                "stderr": "",
                "duration_ms": 5,
                "artifacts": [],
            }

        def fake_llm(prompt, schema_path, **kwargs):
            llm_calls["count"] += 1
            if llm_calls["count"] == 1:
                return json.dumps(
                    {
                        "action": "tool",
                        "tool_name": "ping",
                        "args": {},
                        "action_class": "NETWORK",
                        "timeout_seconds": 3,
                        "response": None,
                    }
                )
            return json.dumps({"action": "respond", "response": "done", "tool_name": None, "args": None, "action_class": None, "timeout_seconds": None})

        assistant = AssistantCore(bridge_json_runner=fake_bridge, llm_text_runner=fake_llm)
        result = assistant.handle_user_input("ping in the range of 192.168.7.2 - 40")
        self.assertEqual(result["response"], "done")
        self.assertEqual(len(calls), 16)
        self.assertEqual(calls[0]["args"]["target"], "192.168.7.2")
        self.assertEqual(calls[-1]["args"]["target"], "192.168.7.17")

    def test_ping_multiple_targets_runs_batch(self):
        calls: list[dict] = []
        llm_calls = {"count": 0}

        def fake_bridge(args, payload):
            if args == ["--list-tools"]:
                return [
                    {
                        "name": "ping",
                        "version": "1.0.0",
                        "description": "desc",
                        "action_class": "NETWORK",
                        "args_schema": {"type": "object"},
                    }
                ]
            calls.append(payload)
            return {
                "ok": True,
                "exit_code": 0,
                "stdout": "pong",
                "stderr": "",
                "duration_ms": 5,
                "artifacts": [],
            }

        def fake_llm(prompt, schema_path, **kwargs):
            llm_calls["count"] += 1
            if llm_calls["count"] == 1:
                return json.dumps(
                    {
                        "action": "tool",
                        "tool_name": "ping",
                        "args": {},
                        "action_class": "NETWORK",
                        "timeout_seconds": 3,
                        "response": None,
                    }
                )
            return json.dumps({"action": "respond", "response": "done", "tool_name": None, "args": None, "action_class": None, "timeout_seconds": None})

        assistant = AssistantCore(bridge_json_runner=fake_bridge, llm_text_runner=fake_llm)
        result = assistant.handle_user_input("ping 8.8.8.8 and 8.8.4.4")
        self.assertEqual(result["response"], "done")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["args"]["target"], "8.8.8.8")
        self.assertEqual(calls[1]["args"]["target"], "8.8.4.4")

    @patch("src.assistant_core.init_tool", return_value={"ok": True, "processed": 3, "skipped": 1})
    def test_add_tool_command_onboards_and_refreshes_tools(self, init_tool_mock):
        calls = {"list_tools": 0}

        def fake_bridge(args, payload):
            if args == ["--list-tools"]:
                calls["list_tools"] += 1
                return [
                    {
                        "name": "ping",
                        "version": "1.0.0",
                        "description": "desc",
                        "action_class": "NETWORK",
                        "args_schema": {"type": "object"},
                    }
                ]
            raise AssertionError("unexpected bridge call")

        def fake_llm(prompt, schema_path, **kwargs):
            raise AssertionError("LLM should not be called for add tool command")

        assistant = AssistantCore(bridge_json_runner=fake_bridge, llm_text_runner=fake_llm)
        result = assistant.handle_user_input("add tool nmap")
        self.assertIn("Tool 'nmap' added.", result["response"])
        self.assertEqual(calls["list_tools"], 2)
        init_tool_mock.assert_called_once_with("nmap")

    @patch("src.assistant_core.init_tool", return_value={"ok": True, "processed": 2, "skipped": 0})
    def test_add_the_tool_phrase_is_detected(self, init_tool_mock):
        calls = {"list_tools": 0}

        def fake_bridge(args, payload):
            if args == ["--list-tools"]:
                calls["list_tools"] += 1
                return [
                    {
                        "name": "ping",
                        "version": "1.0.0",
                        "description": "desc",
                        "action_class": "NETWORK",
                        "args_schema": {"type": "object"},
                    }
                ]
            raise AssertionError("unexpected bridge call")

        assistant = AssistantCore(bridge_json_runner=fake_bridge, llm_text_runner=lambda *a, **k: "")
        result = assistant.handle_user_input("add the tool nmap")
        self.assertIn("Tool 'nmap' added.", result["response"])
        self.assertEqual(calls["list_tools"], 2)
        init_tool_mock.assert_called_once_with("nmap")

    @patch("src.assistant_core.init_tool", return_value={"ok": True, "processed": 1, "skipped": 0})
    def test_unknown_tool_decision_auto_onboards(self, init_tool_mock):
        llm_calls = {"count": 0}

        def fake_bridge(args, payload):
            if args == ["--list-tools"]:
                # First load only ping, second load includes nmap after onboarding.
                if fake_bridge.calls == 0:
                    fake_bridge.calls += 1
                    return [
                        {
                            "name": "ping",
                            "version": "1.0.0",
                            "description": "desc",
                            "action_class": "NETWORK",
                            "args_schema": {"type": "object"},
                        }
                    ]
                return [
                    {
                        "name": "ping",
                        "version": "1.0.0",
                        "description": "desc",
                        "action_class": "NETWORK",
                        "args_schema": {"type": "object"},
                    },
                    {
                        "name": "nmap",
                        "version": "7.0",
                        "description": "scan",
                        "action_class": "SYSTEM",
                        "args_schema": {"type": "object"},
                    },
                ]
            return {
                "ok": True,
                "exit_code": 0,
                "stdout": "scan ok",
                "stderr": "",
                "duration_ms": 5,
                "artifacts": [],
            }

        fake_bridge.calls = 0

        def fake_llm(prompt, schema_path, **kwargs):
            if "assistant_decision.schema.json" in str(schema_path):
                llm_calls["count"] += 1
                if llm_calls["count"] > 1:
                    return json.dumps(
                        {"action": "respond", "response": "Scan complete.", "tool_name": None, "args": None, "action_class": None, "timeout_seconds": None}
                    )
                return json.dumps(
                    {
                        "action": "tool",
                        "tool_name": "nmap",
                        "args": {"_positional": ["192.168.7.3"]},
                        "action_class": "SYSTEM",
                        "timeout_seconds": 3,
                        "response": None,
                    }
                )
            return json.dumps({"response": "Scan complete."})

        assistant = AssistantCore(bridge_json_runner=fake_bridge, llm_text_runner=fake_llm)
        result = assistant.handle_user_input("use nmap on 192.168.7.3")
        self.assertEqual(result["response"], "Scan complete.")
        self.assertEqual(result["tool_call"]["tool_name"], "nmap")
        init_tool_mock.assert_called_once_with("nmap")

    def test_legacy_stop_action_is_normalized_to_respond(self):
        def fake_bridge(args, payload):
            if args == ["--list-tools"]:
                return [
                    {
                        "name": "ping",
                        "version": "1.0.0",
                        "description": "desc",
                        "action_class": "NETWORK",
                        "args_schema": {"type": "object"},
                    }
                ]
            return {
                "ok": True,
                "exit_code": 0,
                "stdout": "ok",
                "stderr": "",
                "duration_ms": 5,
                "artifacts": [],
            }

        def fake_llm(prompt, schema_path, **kwargs):
            return json.dumps({"action": "stop", "reason": "completed"})

        assistant = AssistantCore(bridge_json_runner=fake_bridge, llm_text_runner=fake_llm)
        result = assistant.handle_user_input("hi")
        self.assertEqual(result["response"], "completed")

    def test_nested_decision_packet_is_normalized(self):
        def fake_bridge(args, payload):
            if args == ["--list-tools"]:
                return [
                    {
                        "name": "ping",
                        "version": "1.0.0",
                        "description": "desc",
                        "action_class": "NETWORK",
                        "args_schema": {"type": "object"},
                    }
                ]
            return {
                "ok": True,
                "exit_code": 0,
                "stdout": "ok",
                "stderr": "",
                "duration_ms": 5,
                "artifacts": [],
            }

        def fake_llm(prompt, schema_path, **kwargs):
            return json.dumps({"decision": {"action": "finish", "reason": "all done"}})

        assistant = AssistantCore(bridge_json_runner=fake_bridge, llm_text_runner=fake_llm)
        result = assistant.handle_user_input("hi")
        self.assertEqual(result["response"], "all done")

    def test_repeated_same_tool_call_triggers_guard(self):
        def fake_bridge(args, payload):
            if args == ["--list-tools"]:
                return [
                    {
                        "name": "nmap",
                        "version": "7.0",
                        "description": "scan",
                        "action_class": "SYSTEM",
                        "args_schema": {"type": "object"},
                    }
                ]
            return {
                "ok": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": "failed",
                "duration_ms": 5,
                "artifacts": [],
                "command_preview": "nmap 192.168.7.3",
            }

        llm_calls = {"decision": 0, "final": 0}

        def fake_llm(prompt, schema_path, **kwargs):
            path = str(schema_path)
            if "assistant_decision.schema.json" in path:
                llm_calls["decision"] += 1
                return json.dumps(
                    {
                        "action": "tool",
                        "tool_name": "nmap",
                        "args": {"target": "192.168.7.3"},
                        "action_class": "SYSTEM",
                        "timeout_seconds": 3,
                        "response": None,
                    }
                )
            llm_calls["final"] += 1
            return json.dumps({"response": "nmap failed after retries"})

        assistant = AssistantCore(bridge_json_runner=fake_bridge, llm_text_runner=fake_llm)
        result = assistant.handle_user_input("use nmap on 192.168.7.3")
        self.assertEqual(result["response"], "nmap failed after retries")
        self.assertIn("repeat-loop guard", " | ".join(result.get("logs", [])))
        self.assertEqual(llm_calls["decision"], 2)
        self.assertEqual(llm_calls["final"], 1)

    def test_final_summary_timeout_falls_back_to_deterministic_summary(self):
        def fake_bridge(args, payload):
            if args == ["--list-tools"]:
                return [
                    {
                        "name": "nmap",
                        "version": "7.0",
                        "description": "scan",
                        "action_class": "SYSTEM",
                        "args_schema": {"type": "object"},
                    }
                ]
            return {
                "ok": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": "scan failed",
                "duration_ms": 5,
                "artifacts": [],
                "command_preview": "nmap 192.168.7.3",
            }

        def fake_llm(prompt, schema_path, **kwargs):
            if "assistant_decision.schema.json" in str(schema_path):
                return json.dumps(
                    {
                        "action": "tool",
                        "tool_name": "nmap",
                        "args": {"target": "192.168.7.3"},
                        "action_class": "SYSTEM",
                        "timeout_seconds": 3,
                        "response": None,
                    }
                )
            raise TimeoutError("timed out")

        assistant = AssistantCore(bridge_json_runner=fake_bridge, llm_text_runner=fake_llm)
        result = assistant.handle_user_input("use nmap on 192.168.7.3")
        self.assertIn("Execution summary:", result["response"])
        self.assertIn("Final LLM summary timed out/failed", result["response"])

    def test_generic_respond_after_tool_runs_is_refined_with_final_summary(self):
        def fake_bridge(args, payload):
            if args == ["--list-tools"]:
                return [
                    {
                        "name": "nmap",
                        "version": "7.0",
                        "description": "scan",
                        "action_class": "SYSTEM",
                        "args_schema": {"type": "object"},
                    }
                ]
            return {
                "ok": True,
                "exit_code": 0,
                "stdout": "scan output",
                "stderr": "",
                "duration_ms": 5,
                "artifacts": [],
                "command_preview": "nmap 192.168.7.3",
            }

        calls = {"decision": 0, "final": 0}

        def fake_llm(prompt, schema_path, **kwargs):
            if "assistant_decision.schema.json" in str(schema_path):
                calls["decision"] += 1
                if calls["decision"] == 1:
                    return json.dumps(
                        {
                            "action": "tool",
                            "tool_name": "nmap",
                            "args": {"target": "192.168.7.3"},
                            "action_class": "SYSTEM",
                            "timeout_seconds": 3,
                            "response": None,
                        }
                    )
                return json.dumps({"action": "respond", "response": "Completed.", "tool_name": None, "args": None, "action_class": None, "timeout_seconds": None})
            calls["final"] += 1
            return json.dumps({"response": "Pentest findings: host reachable; no critical open ports found in this run."})

        assistant = AssistantCore(bridge_json_runner=fake_bridge, llm_text_runner=fake_llm)
        result = assistant.handle_user_input("use nmap on 192.168.7.3")
        self.assertIn("Pentest findings:", result["response"])
        self.assertEqual(calls["final"], 1)


if __name__ == "__main__":
    unittest.main()
