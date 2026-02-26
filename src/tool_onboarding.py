from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path

from src.orchestrator.llm import LLMError, extract_json_object, run_llm_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
SPECS_DIR = TOOLS_DIR / "specs"
CACHE_DIR = TOOLS_DIR / "cache"
REGISTRY_PATH = TOOLS_DIR / "registry.json"
OPTION_SCHEMA_PATH = PROJECT_ROOT / "protocol" / "option_spec.schema.json"

ARG_TYPES = {"bool", "int", "float", "string", "enum", "path", "ip", "cidr", "host", "ports", "unknown"}
LOG = logging.getLogger(__name__)


class ToolOnboardingError(RuntimeError):
    pass


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return value


def _run_checked(cmd: list[str], *, timeout_sec: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )


def capture_help(tool: str) -> tuple[str, str, list[str], str]:
    quoted = shlex.quote(tool)
    lookup = _run_checked(["sh", "-lc", f"command -v {quoted}"])
    if lookup.returncode != 0 or not lookup.stdout.strip():
        raise ToolOnboardingError(f"tool not found: {tool}")
    tool_path = lookup.stdout.strip().splitlines()[0].strip()

    help_cmd = [tool, "-h"]
    help_proc = _run_checked(help_cmd, timeout_sec=30)
    if help_proc.returncode != 0:
        help_cmd = [tool, "--help"]
        help_proc = _run_checked(help_cmd, timeout_sec=30)

    help_text = (help_proc.stdout or "") + (help_proc.stderr or "")
    if not help_text.strip():
        help_cmd = [tool, "--help"]
        help_proc = _run_checked(help_cmd, timeout_sec=30)
        help_text = (help_proc.stdout or "") + (help_proc.stderr or "")
        if not help_text.strip():
            raise ToolOnboardingError(f"failed to capture help text for: {tool}")

    version_proc = _run_checked([tool, "--version"], timeout_sec=10)
    version_text = ((version_proc.stdout or "") + (version_proc.stderr or "")).strip()
    version = version_text.splitlines()[0].strip() if version_text else "unknown"

    return help_text, version, help_cmd, tool_path


def _is_section_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.endswith(":"):
        return True
    letters = re.sub(r"[^A-Za-z]", "", stripped)
    if not letters:
        return False
    return stripped.upper() == stripped


def _is_option_start(line: str) -> bool:
    return bool(re.match(r"^\s+--?[\w]", line))


def parse_option_blocks(help_text: str) -> list[dict]:
    lines = help_text.splitlines()
    blocks: list[dict] = []
    section = "GENERAL"
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        if _is_section_header(stripped):
            section = stripped[:-1] if stripped.endswith(":") else stripped
            idx += 1
            continue

        if _is_option_start(line):
            block_lines = [line.rstrip()]
            idx += 1
            while idx < len(lines):
                next_line = lines[idx]
                next_stripped = next_line.strip()
                if not next_stripped:
                    break
                if _is_section_header(next_stripped):
                    break
                if _is_option_start(next_line):
                    break
                block_lines.append(next_line.rstrip())
                idx += 1

            blocks.append({"section": section, "raw_block": "\n".join(block_lines)})
            continue

        idx += 1

    return blocks


def build_option_prompt(tool: str, version: str, section: str, raw_block: str, source_hash: str) -> str:
    return (
        "You are extracting a CLI option spec.\n\n"
        f"Tool: {tool}\n"
        f"Version: {version}\n"
        f"Section: {section}\n\n"
        "Help block:\n"
        f"{raw_block}\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "flags": ["..."],\n'
        '  "arg_syntax": null | "<arg>" | "[arg]" | "--opt=<arg>" | "--opt <arg>",\n'
        '  "arg_type": "bool" | "int" | "float" | "string" | "enum" | "path" | "ip" | "cidr" | "host" | "ports" | "unknown",\n'
        '  "default": null,\n'
        '  "summary": "...",\n'
        '  "details": "",\n'
        '  "conflicts_with": [],\n'
        '  "implies": [],\n'
        f'  "category": "{section}",\n'
        '  "risk_tags": [],\n'
        f'  "source_hash": "{source_hash}"\n'
        "}"
    )


def _validate_option_spec(spec: dict) -> None:
    required = {
        "flags",
        "arg_syntax",
        "arg_type",
        "default",
        "summary",
        "details",
        "conflicts_with",
        "implies",
        "category",
        "risk_tags",
        "source_hash",
    }
    if not isinstance(spec, dict):
        raise ToolOnboardingError("option spec must be object")
    missing = required - set(spec.keys())
    if missing:
        raise ToolOnboardingError(f"option spec missing keys: {', '.join(sorted(missing))}")
    if not isinstance(spec["flags"], list) or any(not isinstance(v, str) for v in spec["flags"]):
        raise ToolOnboardingError("flags must be string[]")
    if spec["arg_syntax"] is not None and not isinstance(spec["arg_syntax"], str):
        raise ToolOnboardingError("arg_syntax must be string|null")
    if not isinstance(spec["arg_type"], str) or spec["arg_type"] not in ARG_TYPES:
        raise ToolOnboardingError("arg_type invalid")
    if not isinstance(spec["summary"], str):
        raise ToolOnboardingError("summary must be string")
    if not isinstance(spec["details"], str):
        raise ToolOnboardingError("details must be string")
    for name in ("conflicts_with", "implies", "risk_tags"):
        if not isinstance(spec[name], list) or any(not isinstance(v, str) for v in spec[name]):
            raise ToolOnboardingError(f"{name} must be string[]")
    if not isinstance(spec["category"], str):
        raise ToolOnboardingError("category must be string")
    if not isinstance(spec["source_hash"], str):
        raise ToolOnboardingError("source_hash must be string")


def analyze_option(prompt: str) -> dict | None:
    model_path = os.getenv("LOA_MODEL_PATH", "local")
    attempts = [prompt, prompt + "\n\nYour previous output was invalid. Return ONLY a strict JSON object with all required keys and valid types."]

    for candidate in attempts:
        try:
            text = run_llm_text(
                candidate,
                OPTION_SCHEMA_PATH,
                model_path=model_path,
                n_ctx=int(os.getenv("LOA_N_CTX", "2048")),
                temp=0.0,
                seed=int(os.getenv("LOA_SEED", "0")),
                log_dir=None,
            )
            spec = extract_json_object(text)
            _validate_option_spec(spec)
            return spec
        except (LLMError, ToolOnboardingError, ValueError):
            continue
    return None


def get_canonical_key(flags: list[str]) -> str:
    if not flags:
        return "unknown"
    preferred = next((flag for flag in flags if isinstance(flag, str) and flag.startswith("--")), flags[0])
    key = str(preferred).strip()
    key = key.lstrip("-")
    key = key.replace("=", "")
    key = key.replace(" ", "_")
    return key or "unknown"


def _update_registry(tool: str, version: str, tool_path: str) -> None:
    payload = _read_json(REGISTRY_PATH, {"tools": []})
    if not isinstance(payload, dict):
        payload = {"tools": []}
    tools = payload.get("tools")
    if not isinstance(tools, list):
        tools = []

    updated = False
    for item in tools:
        if isinstance(item, dict) and item.get("name") == tool:
            item["version"] = version
            item["path"] = tool_path
            updated = True
            break
    if not updated:
        tools.append({"name": tool, "version": version, "path": tool_path})
    payload["tools"] = tools
    _atomic_write_json(REGISTRY_PATH, payload)


def load_registry() -> dict:
    payload = _read_json(REGISTRY_PATH, {"tools": []})
    if isinstance(payload, dict) and isinstance(payload.get("tools"), list):
        return payload
    return {"tools": []}


def load_tool_spec(tool: str) -> dict:
    path = SPECS_DIR / f"{tool}.json"
    if not path.exists():
        raise ToolOnboardingError(f"tool spec not found: {tool}")
    payload = _read_json(path, {})
    if isinstance(payload, dict):
        return payload
    raise ToolOnboardingError(f"invalid tool spec: {path}")


def init_tool(tool: str) -> dict:
    help_text, version, help_cmd, tool_path = capture_help(tool)
    help_hash = _sha256(help_text + version)

    spec_path = SPECS_DIR / f"{tool}.json"
    cache_path = CACHE_DIR / f"{tool}.json"

    spec = _read_json(
        spec_path,
        {"name": tool, "version": version, "help_cmd": help_cmd, "help_hash": "", "options": {}},
    )
    if not isinstance(spec, dict):
        spec = {"name": tool, "version": version, "help_cmd": help_cmd, "help_hash": "", "options": {}}

    if spec.get("help_hash") == help_hash:
        _update_registry(tool, version, tool_path)
        return {"ok": True, "processed": 0, "skipped": 0}

    options = spec.get("options")
    if not isinstance(options, dict):
        options = {}
    spec["options"] = options
    spec["name"] = tool
    spec["version"] = version
    spec["help_cmd"] = help_cmd
    spec["help_hash"] = help_hash

    blocks = parse_option_blocks(help_text)
    cache = _read_json(cache_path, {})
    if not isinstance(cache, dict):
        cache = {}

    processed = 0
    skipped = 0

    for block in blocks:
        section = block["section"]
        raw_block = block["raw_block"]
        source_hash = _sha256(raw_block + section)

        cached = cache.get(source_hash)
        if isinstance(cached, dict):
            option_spec = cached
        else:
            prompt = build_option_prompt(tool, version, section, raw_block, source_hash)
            option_spec = analyze_option(prompt)
            if option_spec is None:
                LOG.warning("skipping option block for tool=%s source_hash=%s", tool, source_hash)
                skipped += 1
                continue
            option_spec["source_hash"] = source_hash
            cache[source_hash] = option_spec
            _atomic_write_json(cache_path, cache)

        key = get_canonical_key(option_spec.get("flags", []))
        spec["options"][key] = option_spec
        processed += 1
        _atomic_write_json(spec_path, spec)

    _atomic_write_json(spec_path, spec)
    _atomic_write_json(cache_path, cache)
    _update_registry(tool, version, tool_path)
    return {"ok": True, "processed": processed, "skipped": skipped}
