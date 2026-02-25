import subprocess
import unittest
from unittest.mock import patch

from src.loa_bridge import _dispatch_tool, _list_tools


class TestLoaBridge(unittest.TestCase):
    def test_list_tools_contains_ping(self):
        tools = _list_tools()
        names = {tool["name"] for tool in tools}
        self.assertIn("ping", names)

    @patch("src.loa_bridge.rollback")
    @patch("src.loa_bridge.snapshot", return_value="abc123")
    @patch("src.loa_bridge.subprocess.run")
    def test_dispatch_success(self, run_mock, snapshot_mock, rollback_mock):
        run_mock.return_value = subprocess.CompletedProcess(
            args=["bash", "tools/ping/ping.sh", "8.8.8.8", "1"],
            returncode=0,
            stdout="ok",
            stderr="",
        )
        result = _dispatch_tool(
            {
                "tool_name": "ping",
                "args": {"target": "8.8.8.8", "count": 1},
                "cwd": None,
                "timeout_seconds": 3,
                "action_class": "NETWORK",
                "env": None,
            }
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["exit_code"], 0)
        snapshot_mock.assert_called_once()
        rollback_mock.assert_not_called()

    @patch("src.loa_bridge.rollback")
    @patch("src.loa_bridge.snapshot", return_value="abc123")
    @patch("src.loa_bridge.subprocess.run")
    def test_dispatch_failure_rolls_back(self, run_mock, snapshot_mock, rollback_mock):
        run_mock.return_value = subprocess.CompletedProcess(
            args=["bash", "tools/ping/ping.sh", "8.8.8.8", "1"],
            returncode=1,
            stdout="",
            stderr="failed",
        )
        result = _dispatch_tool(
            {
                "tool_name": "ping",
                "args": {"target": "8.8.8.8", "count": 1},
                "cwd": None,
                "timeout_seconds": 3,
                "action_class": "NETWORK",
                "env": None,
            }
        )
        self.assertFalse(result["ok"])
        rollback_mock.assert_called_once_with("abc123")
        snapshot_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
