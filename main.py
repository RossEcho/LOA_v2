from __future__ import annotations

import argparse
import json
import sys

from src.assistant_core import AssistantCore, AssistantError
from src.loa import main as loa_main
from src.tool_onboarding import ToolOnboardingError, init_tool, load_registry, load_tool_spec


def _run_once(assistant: AssistantCore, message: str) -> int:
    result = assistant.handle_user_input(message)
    for line in result.get("logs", []) or []:
        print(f"log> {line}")
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
        for line in result.get("logs", []) or []:
            print(f"log> {line}")
        print(f"assistant> {result['response']}")


def _run_assistant(args: argparse.Namespace) -> int:
    assistant = AssistantCore()
    if args.once:
        result = assistant.handle_user_input(args.once)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(result["response"])
        return 0
    return _run_repl(assistant)


def _run_loa(args: argparse.Namespace) -> int:
    # Delegate to the orchestrator CLI to keep a single runtime flow via main.py.
    sys.argv = ["loa"] + list(args.loa_args or [])
    return loa_main()


def _run_top_menu() -> int:
    while True:
        print("\nMain Menu")
        print("1) LOA orchestrator menu")
        print("2) Assistant REPL")
        print("3) Exit")
        choice = input("Select [1-3]: ").strip()
        if choice == "1":
            sys.argv = ["loa", "menu"]
            code = loa_main()
            if code != 0:
                print(f"loa exited with code {code}")
            continue
        if choice == "2":
            try:
                return _run_assistant(argparse.Namespace(once=None, json=False))
            except AssistantError as exc:
                print(f"assistant error: {exc}", file=sys.stderr)
                return 1
        if choice == "3":
            return 0
        print("Invalid selection")


def main() -> int:
    parser = argparse.ArgumentParser(prog="main", description="Unified entrypoint for LOA and assistant flows")
    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("menu", help="Top-level menu (LOA menu or assistant REPL)")

    assistant_cmd = sub.add_parser("assistant", help="Assistant mode")
    assistant_cmd.add_argument("--once", help="Single user prompt to process")
    assistant_cmd.add_argument("--json", action="store_true", help="Print full response payload as JSON")
    assistant_cmd.add_argument("--init-tool", help="Initialize tool spec from CLI help output")
    assistant_cmd.add_argument("--list-tools", action="store_true", help="List onboarded tools from registry")
    assistant_cmd.add_argument("--tool-spec", help="Print full spec JSON for a tool")

    loa_cmd = sub.add_parser("loa", help="Pass-through to LOA orchestrator CLI")
    loa_cmd.add_argument("loa_args", nargs=argparse.REMAINDER, help="Arguments forwarded to src/loa.py")

    args = parser.parse_args()

    if args.command in (None, "menu"):
        return _run_top_menu()

    if args.command == "loa":
        return _run_loa(args)

    if args.command == "assistant":
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
            return _run_assistant(args)
        except AssistantError as exc:
            print(f"assistant error: {exc}", file=sys.stderr)
            return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
