from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


class LLMError(RuntimeError):
    pass


def _extract_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            trailing = text[idx + end :].strip()
            if not trailing:
                return obj
    raise LLMError("model output did not contain a valid JSON object")


def _run_cli(argv: list[str], prompt: str, timeout_sec: int) -> str:
    proc = subprocess.run(
        argv,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise LLMError(f"llm command failed: {msg}")
    output = proc.stdout.strip()
    if not output:
        raise LLMError("llm produced empty output")
    return output


def run_llm_json(
    prompt: str,
    schema_path: str | Path,
    *,
    model_path: str,
    n_ctx: int,
    temp: float,
    seed: int,
) -> dict:
    schema = Path(schema_path).resolve()
    bin_path = Path(os.path.expanduser(os.getenv("LOA_LLAMA_CLI_BIN", "llama-cli")))
    timeout_sec = int(os.getenv("LOA_LLM_TIMEOUT_SEC", "90"))

    base_cmd = [
        str(bin_path),
        "-m",
        model_path,
        "--temp",
        str(temp),
        "--seed",
        str(seed),
        "--ctx-size",
        str(n_ctx),
        "-n",
        "512",
        "-f",
        "-",
    ]

    schema_flag = os.getenv("LOA_LLAMA_SCHEMA_FLAG", "--json-schema")
    cmd_with_schema = base_cmd + [schema_flag, str(schema)]

    try:
        raw = _run_cli(cmd_with_schema, prompt=prompt, timeout_sec=timeout_sec)
    except Exception:
        raw = _run_cli(base_cmd, prompt=prompt, timeout_sec=timeout_sec)

    return _extract_json_object(raw)
