import unittest

from src.orchestrator.agent import (
    AgentLoopError,
    _normalize_decision_packet,
    _normalize_next_step,
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
        validate_decision({"action": "stop", "reason": "done"})

    def test_decision_reason_is_coerced_not_rejected(self):
        decision, note = _normalize_decision_reason({"action": "stop", "reason": {"x": 1}})
        self.assertIsNotNone(note)
        self.assertIsInstance(decision["reason"], str)
        validate_decision(decision)

    def test_normalize_next_step_maps_optional_timeout(self):
        step, notes = _normalize_next_step(
            {"id": "s1", "tool": "ping", "args": {"target": "8.8.8.8"}, "optional_timeout_sec": 30},
            original_prompt="ping 8.8.8.8",
            iteration=3,
        )
        self.assertTrue(notes)
        self.assertEqual(step.get("timeout_sec"), 30)
        self.assertNotIn("optional_timeout_sec", step)

    def test_normalize_next_step_repairs_missing_ping_target(self):
        step, notes = _normalize_next_step(
            {"tool": "ping", "args": {}},
            original_prompt="please ping 8.8.8.8",
            iteration=1,
        )
        self.assertTrue(any("repaired" in note for note in notes))
        self.assertEqual(step["args"]["target"], "8.8.8.8")

    def test_normalize_next_step_rejects_unparseable(self):
        with self.assertRaises(AgentLoopError):
            _normalize_next_step({"foo": "bar"}, original_prompt="hello", iteration=1)

    def test_normalize_decision_packet_maps_legacy_continue(self):
        packet, diag = _normalize_decision_packet(
            {"action": "continue", "next_plan": {"tool": "ping", "args": {"target": "8.8.8.8"}}},
            original_prompt="ping 8.8.8.8",
            iteration=2,
        )
        self.assertEqual(packet["decision"]["action"], "run_tool")
        self.assertIsNotNone(packet["next_step"])
        self.assertIn("action_note", diag)


if __name__ == "__main__":
    unittest.main()
