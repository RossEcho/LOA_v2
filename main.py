from __future__ import annotations

import argparse
import json
import sys

from src.assistant_core import AssistantCore, AssistantError
from src.tool_onboarding import ToolOnboardingError, init_tool, load_registry, load_tool_spec


def _run_once(assistant: AssistantCore, message: str) -> int:
    result = assistant.handle_user_input(message)
    print(result["response"])
    return 0


def _run_repl(assistant: AssistantCore) -> int:
    while True:
        try:
            message = input("you> ").strip()
        except EOFError:
            return 0
        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            return 0
        result = assistant.handle_user_input(message)
        print(f"assistant> {result['response']}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="loa-assistant", description="Minimal assistant layer over LOA bridge")
    parser.add_argument("--once", help="Single user prompt to process")
    parser.add_argument("--json", action="store_true", help="Print full response payload as JSON")
    parser.add_argument("--init-tool", help="Initialize tool spec from CLI help output")
    parser.add_argument("--list-tools", action="store_true", help="List onboarded tools from registry")
    parser.add_argument("--tool-spec", help="Print full spec JSON for a tool")
    args = parser.parse_args()

    if args.list_tools:
        print(json.dumps(load_registry().get("tools", []), ensure_ascii=False))
        return 0

    if args.tool_spec:
        try:
            print(json.dumps(load_tool_spec(args.tool_spec), ensure_ascii=False))
            return 0
        except ToolOnboardingError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 1

    if args.init_tool:
        try:
            print(json.dumps(init_tool(args.init_tool), ensure_ascii=False))
            return 0
        except ToolOnboardingError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 1

    try:
        assistant = AssistantCore()
        if args.once:
            result = assistant.handle_user_input(args.once)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(result["response"])
            return 0
        return _run_repl(assistant)
    except AssistantError as exc:
        print(f"assistant error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
