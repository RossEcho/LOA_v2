import subprocess
import unittest
from unittest.mock import patch

from src import savestate_git


class TestSaveStateGit(unittest.TestCase):
    @patch("src.savestate_git.subprocess.run")
    def test_snapshot_with_changes_commits(self, run_mock):
        run_mock.side_effect = [
            subprocess.CompletedProcess(args=["git", "add", "-A"], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(
                args=["git", "diff", "--cached", "--quiet"], returncode=1, stdout="", stderr=""
            ),
            subprocess.CompletedProcess(
                args=["git", "commit", "-m", "checkpoint: test"], returncode=0, stdout="", stderr=""
            ),
            subprocess.CompletedProcess(args=["git", "rev-parse", "HEAD"], returncode=0, stdout="abc\n", stderr=""),
        ]
        commit = savestate_git.snapshot("test")
        self.assertEqual(commit, "abc")

    @patch("src.savestate_git.subprocess.run")
    def test_rollback_excludes_persisted_dirs(self, run_mock):
        run_mock.side_effect = [
            subprocess.CompletedProcess(args=["git", "reset", "--hard", "abc"], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(
                args=[
                    "git",
                    "clean",
                    "-fd",
                    "-e",
                    ".memory",
                    "-e",
                    ".secrets",
                    "-e",
                    ".logs",
                ],
                returncode=0,
                stdout="",
                stderr="",
            ),
        ]
        savestate_git.rollback("abc")
        self.assertEqual(run_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
