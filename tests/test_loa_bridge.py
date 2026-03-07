import subprocess
import unittest
from unittest.mock import patch

from src.loa_bridge import _dispatch_tool, _list_tools


class TestLoaBridge(unittest.TestCase):
    def test_list_tools_contains_ping(self):
        tools = _list_tools()
        names = {tool["name"] for tool in tools}
        self.assertIn("ping", names)

    @patch("src.loa_bridge.load_registry", return_value={"tools": [{"name": "python", "version": "3.12", "path": "/usr/bin/python"}]})
    def test_list_tools_includes_onboarded(self, _registry_mock):
        tools = _list_tools()
        names = {tool["name"] for tool in tools}
        self.assertIn("python", names)

    @patch(
        "src.loa_bridge.load_registry",
        return_value={
            "tools": [
                {
                    "name": "python",
                    "version": "3.12",
                    "path": "/usr/bin/python",
                    "description": "Python interpreter",
                    "usage": "python [options] script.py",
                }
            ]
        },
    )
    def test_list_tools_uses_registry_description_and_usage(self, _registry_mock):
        tools = _list_tools()
        python_tool = next(tool for tool in tools if tool["name"] == "python")
        self.assertEqual(python_tool["description"], "Python interpreter")
        self.assertEqual(python_tool["usage"], "python [options] script.py")

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
        self.assertIn("command_preview", result)
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

    @patch("src.loa_bridge.load_tool_spec", return_value={"options": {}})
    @patch("src.loa_bridge.load_registry", return_value={"tools": [{"name": "nmap", "version": "7.9", "path": "/usr/bin/nmap"}]})
    @patch("src.loa_bridge.rollback")
    @patch("src.loa_bridge.snapshot", return_value="abc123")
    @patch("src.loa_bridge.subprocess.run")
    def test_dispatch_onboarded_unknown_args_fall_back_to_positional(
        self, run_mock, snapshot_mock, rollback_mock, _registry_mock, _spec_mock
    ):
        run_mock.return_value = subprocess.CompletedProcess(
            args=["/usr/bin/nmap", "192.168.7.3"],
            returncode=0,
            stdout="ok",
            stderr="",
        )
        result = _dispatch_tool(
            {
                "tool_name": "nmap",
                "args": {"scan_type": "host", "target": "192.168.7.3"},
                "cwd": None,
                "timeout_seconds": 3,
                "action_class": "SYSTEM",
                "env": None,
            }
        )
        self.assertTrue(result["ok"])
        called_argv = run_mock.call_args[0][0]
        self.assertEqual(called_argv[0], "/usr/bin/nmap")
        self.assertIn("192.168.7.3", called_argv)
        self.assertNotIn("--scan-type", called_argv)
        rollback_mock.assert_not_called()
        snapshot_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
