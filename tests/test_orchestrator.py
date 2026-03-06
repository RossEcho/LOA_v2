import tempfile
import unittest
from pathlib import Path

from src.orchestrator.executor import build_meta, create_session_dir, execute_plan, write_plan_and_meta
from src.orchestrator.planner import PlanValidationError, validate_plan
from src.orchestrator.tools import ToolValidationError, validate_tool_args


class TestOrchestratorPlannerValidation(unittest.TestCase):
    def test_plan_schema_required_fields(self):
        with self.assertRaises(PlanValidationError):
            validate_plan({"steps": []}, for_execution=False)

    def test_unknown_tool_rejected(self):
        plan = {
            "plan_id": "p1",
            "session_name": "x",
            "steps": [{"id": "s1", "tool": "unknown", "args": {}}],
            "final_output": "final.json",
        }
        with self.assertRaises(PlanValidationError):
            validate_plan(plan, for_execution=True)

    def test_disabled_tool_rejected_for_execution(self):
        plan = {
            "plan_id": "p1",
            "session_name": "x",
            "steps": [{"id": "s1", "tool": "SmarTar", "args": {"db": "a", "archive": "b", "query": "c"}}],
            "final_output": "final.json",
        }
        with self.assertRaises(PlanValidationError):
            validate_plan(plan, for_execution=True)

    def test_outputs_field_is_stripped_for_backward_compat(self):
        plan = {
            "plan_id": "p3",
            "session_name": "x",
            "steps": [
                {
                    "id": "s1",
                    "tool": "ping",
                    "args": {"target": "8.8.8.8"},
                    "outputs": {"latency_ms": "42"},
                }
            ],
            "final_output": "final.json",
        }
        validate_plan(plan, for_execution=False)
        self.assertNotIn("outputs", plan["steps"][0])

    def test_outputs_artifact_keys_are_ignored_and_removed(self):
        plan = {
            "plan_id": "p4",
            "session_name": "x",
            "steps": [
                {
                    "id": "s1",
                    "tool": "ping",
                    "args": {"target": "8.8.8.8"},
                    "outputs": {
                        "stdout_file": "steps/step_001_ping/stdout.txt",
                        "stderr_file": "steps/step_001_ping/stderr.txt",
                        "exit_code_file": "steps/step_001_ping/exit_code.txt",
                        "timing_file": "steps/step_001_ping/timing.json",
                    },
                }
            ],
            "final_output": "final.json",
        }
        validate_plan(plan, for_execution=False)
        self.assertNotIn("outputs", plan["steps"][0])

    def test_unknown_dependency_rejected(self):
        plan = {
            "plan_id": "p_dep",
            "session_name": "x",
            "steps": [
                {
                    "id": "s1",
                    "tool": "ping",
                    "args": {"target": "8.8.8.8"},
                    "depends_on": ["missing"],
                }
            ],
            "final_output": "final.json",
        }
        with self.assertRaises(PlanValidationError):
            validate_plan(plan, for_execution=False)


class TestToolRegistryValidation(unittest.TestCase):
    def test_ping_args_validation(self):
        validate_tool_args("ping", {"target": "example.com", "count": 2})
        with self.assertRaises(ToolValidationError):
            validate_tool_args("ping", {"target": "", "count": 2})
        with self.assertRaises(ToolValidationError):
            validate_tool_args("ping", {"target": "example.com", "count": 99})


class TestExecutorStructure(unittest.TestCase):
    def test_session_structure_created(self):
        plan = {
            "plan_id": "p2",
            "session_name": "session",
            "steps": [],
            "final_output": "final.json",
        }
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp) / "runs"
            session_dir = create_session_dir("session", runs_root=runs_root)
            meta = build_meta(model_path="~/models/qwen.gguf", n_ctx=2048, temp=0.0, seed=0)
            write_plan_and_meta(session_dir, plan, meta)
            result = execute_plan(plan, session_dir)

            self.assertTrue((session_dir / "plan.json").exists())
            self.assertTrue((session_dir / "meta.json").exists())
            self.assertTrue((session_dir / "steps").exists())
            self.assertTrue((session_dir / "final.json").exists())
            self.assertEqual(result["steps_executed"], 0)
            self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main()
