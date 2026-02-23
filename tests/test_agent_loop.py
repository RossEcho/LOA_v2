import unittest

from src.orchestrator.agent import (
    AgentLoopError,
    _normalize_next_plan,
    _normalize_decision_reason,
    validate_analysis,
    validate_decision,
)


class TestAgentValidators(unittest.TestCase):
    def test_validate_analysis_ok(self):
        validate_analysis(
            {
                "summary": "ok",
                "observations": ["o1"],
                "errors": [],
                "confidence": 0.8,
            }
        )

    def test_validate_analysis_bad_confidence(self):
        with self.assertRaises(AgentLoopError):
            validate_analysis(
                {
                    "summary": "x",
                    "observations": [],
                    "errors": [],
                    "confidence": 2,
                }
            )

    def test_validate_decision_continue_requires_next_plan(self):
        with self.assertRaises(AgentLoopError):
            validate_decision({"action": "continue", "reason": "need more"})

    def test_validate_decision_finish_ok(self):
        validate_decision({"action": "finish", "reason": "done"})

    def test_decision_reason_is_coerced_not_rejected(self):
        decision, note = _normalize_decision_reason({"action": "finish", "reason": {"x": 1}})
        self.assertIsNotNone(note)
        self.assertIsInstance(decision["reason"], str)
        validate_decision(decision)

    def test_normalize_next_plan_wraps_step_like(self):
        plan, note = _normalize_next_plan({"tool": "ping", "args": {"target": "8.8.8.8"}}, iteration=2)
        self.assertIsNotNone(note)
        self.assertIn("steps", plan)
        self.assertEqual(len(plan["steps"]), 1)
        self.assertEqual(plan["steps"][0]["tool"], "ping")
        self.assertEqual(plan["final_output"], "final.json")

    def test_normalize_next_plan_maps_optional_timeout(self):
        plan, note = _normalize_next_plan(
            {"id": "s1", "tool": "ping", "args": {"target": "8.8.8.8"}, "optional_timeout_sec": 30},
            iteration=3,
        )
        self.assertIsNotNone(note)
        step = plan["steps"][0]
        self.assertEqual(step.get("timeout_sec"), 30)
        self.assertNotIn("optional_timeout_sec", step)

    def test_normalize_next_plan_rejects_unparseable(self):
        with self.assertRaises(AgentLoopError):
            _normalize_next_plan({"foo": "bar"}, iteration=1)


if __name__ == "__main__":
    unittest.main()
