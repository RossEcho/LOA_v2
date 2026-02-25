import json
import unittest

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
            return json.dumps({"response": "Ping completed."})

        assistant = AssistantCore(bridge_json_runner=fake_bridge, llm_text_runner=fake_llm)
        result = assistant.handle_user_input("ping 8.8.8.8")
        self.assertEqual(result["response"], "Ping completed.")
        self.assertEqual(result["tool_call"]["tool_name"], "ping")
        self.assertTrue(result["tool_result"]["ok"])

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


if __name__ == "__main__":
    unittest.main()
