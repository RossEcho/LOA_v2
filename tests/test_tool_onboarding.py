import unittest
from pathlib import Path
import shutil
from unittest.mock import patch

from src import tool_onboarding


class TestToolOnboarding(unittest.TestCase):
    def test_parse_option_blocks(self):
        help_text = """
OPTIONS:
  -h, --help            show help
  -p, --port <port>     set port
    continuation line

ADVANCED:
  --fast                speed mode
""".strip(
            "\n"
        )
        blocks = tool_onboarding.parse_option_blocks(help_text)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[0]["section"], "OPTIONS")
        self.assertIn("--port", blocks[1]["raw_block"])
        self.assertEqual(blocks[2]["section"], "ADVANCED")

    def test_get_canonical_key_prefers_long_flag(self):
        key = tool_onboarding.get_canonical_key(["-p", "--port=<port>"])
        self.assertEqual(key, "port<port>")

    def test_init_tool_writes_spec_and_cache(self):
        root = Path("tests") / ".tmp_tool_onboarding"
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        try:
            specs_dir = root / "specs"
            cache_dir = root / "cache"
            registry_path = root / "registry.json"

            with (
                patch.object(tool_onboarding, "SPECS_DIR", specs_dir),
                patch.object(tool_onboarding, "CACHE_DIR", cache_dir),
                patch.object(tool_onboarding, "REGISTRY_PATH", registry_path),
                patch.object(
                    tool_onboarding,
                    "capture_help",
                    return_value=("OPTIONS:\n  -h, --help show help\n", "1.0.0", ["demo", "-h"], "/usr/bin/demo"),
                ),
                patch.object(
                    tool_onboarding,
                    "analyze_option",
                    return_value={
                        "flags": ["-h", "--help"],
                        "arg_syntax": None,
                        "arg_type": "bool",
                        "default": None,
                        "summary": "help",
                        "details": "",
                        "conflicts_with": [],
                        "implies": [],
                        "category": "OPTIONS",
                        "risk_tags": [],
                        "source_hash": "placeholder",
                    },
                ),
            ):
                result = tool_onboarding.init_tool("demo")
                self.assertTrue(result["ok"])
                self.assertEqual(result["processed"], 1)

                spec_path = specs_dir / "demo.json"
                cache_path = cache_dir / "demo.json"
                self.assertTrue(spec_path.exists())
                self.assertTrue(cache_path.exists())
                self.assertTrue(registry_path.exists())
                registry = tool_onboarding._read_json(registry_path, {})
                self.assertIn("tools", registry)
                self.assertEqual(registry["tools"][0]["name"], "demo")
                self.assertIn("description", registry["tools"][0])
                self.assertIn("usage", registry["tools"][0])

                cached_result = tool_onboarding.init_tool("demo")
                self.assertTrue(cached_result["ok"])
                self.assertEqual(cached_result["processed"], 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
