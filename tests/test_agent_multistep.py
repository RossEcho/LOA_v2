import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.orchestrator.agent import run_agent_loop


class TestAgentMultiStep(unittest.TestCase):
    def test_multi_step_execution_with_continuation_append(self):
        initial_plan = {
            "plan_id": "p_multi",
            "session_name": "multi",
            "steps": [
                {
                    "id": "s1",
                    "description": "First ping",
                    "tool": "ping",
                    "args": {"target": "8.8.8.8", "count": 1},
                    "depends_on": [],
                    "success_criteria": {"type": "exit_code", "equals": 0},
                    "retry_policy": {"max_retries": 0},
                },
                {
                    "id": "s2",
                    "description": "Second ping",
                    "tool": "ping",
                    "args": {"target": "1.1.1.1", "count": 1},
                    "depends_on": ["s1"],
                    "success_criteria": {"type": "exit_code", "equals": 0},
                    "retry_policy": {"max_retries": 0},
                },
            ],
            "final_output": "final.json",
        }

        continuation_payloads = [
            json.dumps(
                {
                    "append_steps": [
                        {
                            "id": "s3",
                            "description": "Continuation ping",
                            "tool": "ping",
                            "args": {"target": "9.9.9.9", "count": 1},
                            "depends_on": ["s2"],
                            "success_criteria": {"type": "exit_code", "equals": 0},
                            "retry_policy": {"max_retries": 0},
                        }
                    ],
                    "reason": "Need final verification",
                }
            ),
            json.dumps({"append_steps": [], "reason": "No further work"}),
        ]

        def fake_run_llm_text(*args, **kwargs):
            if continuation_payloads:
                return continuation_payloads.pop(0)
            return json.dumps({"append_steps": [], "reason": "done"})

        def fake_execute_plan(plan: dict, session_dir: Path) -> dict:
            step = plan["steps"][0]
            out_dir = session_dir / "fake_step"
            out_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = out_dir / "stdout.txt"
            stderr_path = out_dir / "stderr.txt"
            stdout_path.write_text(f"ok:{step['id']}", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            result = {
                "id": step["id"],
                "tool": step["tool"],
                "exit_code": 0,
                "timed_out": False,
                "elapsed_sec": 0.01,
                "stdout_file": str(stdout_path),
                "stderr_file": str(stderr_path),
            }
            return {
                "plan_id": plan["plan_id"],
                "session_name": plan["session_name"],
                "success": True,
                "notes": None,
                "steps_total": 1,
                "steps_executed": 1,
                "elapsed_sec": 0.01,
                "results": [result],
                "completed_at": "2026-01-01T00:00:00+00:00",
            }

        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "session"
            session_dir.mkdir(parents=True, exist_ok=True)

            with patch("src.orchestrator.agent.run_llm_text", side_effect=fake_run_llm_text):
                with patch("src.orchestrator.agent.execute_plan", side_effect=fake_execute_plan):
                    summary = run_agent_loop(
                        session_dir=session_dir,
                        original_prompt="validate connectivity in several steps",
                        initial_plan=initial_plan,
                        meta={"test": True},
                        model_path="/tmp/model.gguf",
                        n_ctx=2048,
                        temp=0.0,
                        seed=0,
                        max_steps=6,
                        multi_step_mode=True,
                        max_retries=0,
                        max_expansions=2,
                        max_replans=0,
                        max_runtime_sec=None,
                    )

            self.assertEqual(summary["mode"], "multi_step")
            self.assertEqual(summary["summary"], "success")
            self.assertEqual(summary["iterations"], 3)

            session_state = json.loads((session_dir / "session_state.json").read_text(encoding="utf-8"))
            self.assertIn("s3", [step["id"] for step in session_state["current_plan"]["steps"]])
            self.assertEqual(len(session_state["completed_steps"]), 3)
            self.assertEqual(session_state["continuation_expansions"], 2)


if __name__ == "__main__":
    unittest.main()
