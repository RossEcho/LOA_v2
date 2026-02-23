import unittest

from src.orchestrator.agent import AgentLoopError, validate_analysis, validate_decision


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


if __name__ == "__main__":
    unittest.main()
