from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
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


def _run_cli(argv: list[str], timeout_sec: int) -> str:
    blocked = {"-i", "--interactive-first", "-r"}
    if any(flag in argv for flag in blocked):
        raise LLMError("interactive llama-cli flags are not allowed")

    proc = subprocess.run(
        argv,
        text=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
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


def _run_server(prompt: str, schema_path: Path, timeout_sec: int, temp: float, seed: int) -> str:
    endpoint = os.getenv("LOA_LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1/chat/completions")
    max_tokens = int(os.getenv("LOA_LLM_MAX_TOKENS", "512"))
    model_name = os.getenv("LOA_LLAMA_SERVER_MODEL", "local")

    payload: dict = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
    }

    # Optional OpenAI-style response_format; many llama-server builds ignore unknown fields.
    if os.getenv("LOA_LLM_SERVER_USE_SCHEMA", "0") == "1":
        try:
            schema_obj = json.loads(schema_path.read_text(encoding="utf-8"))
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "loa_plan", "schema": schema_obj},
            }
        except Exception:
            pass

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise LLMError(f"llm server request failed: {exc}") from exc

    try:
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise LLMError("llm server response not in chat completions format") from exc

    if not isinstance(content, str) or not content.strip():
        raise LLMError("llm server returned empty content")

    return content


def _run_cli_json(prompt: str, schema_path: Path, model_path: str, n_ctx: int, temp: float, seed: int) -> str:
    bin_path = Path(os.path.expanduser(os.getenv("LOA_LLAMA_CLI_BIN", "llama-cli")))
    timeout_sec = int(os.getenv("LOA_LLM_TIMEOUT_SEC", "90"))
    prompt_max_chars = int(os.getenv("LOA_LLM_PROMPT_MAX_CHARS", "4000"))

    base_cmd = [
        str(bin_path),
        "-m",
        model_path,
        os.getenv("LOA_LLAMA_NO_CNV_FLAG", "-no-cnv"),
        "--temp",
        str(temp),
        "--seed",
        str(seed),
        "--ctx-size",
        str(n_ctx),
        "-n",
        os.getenv("LOA_LLM_MAX_TOKENS", "512"),
    ]

    schema_flag = os.getenv("LOA_LLAMA_SCHEMA_FLAG", "--json-schema")
    prompt_flag = os.getenv("LOA_LLAMA_PROMPT_FLAG", "-p")

    cleanup = None
    if len(prompt) <= prompt_max_chars:
        cmd = base_cmd + [prompt_flag, prompt]
    else:
        temp_dir = tempfile.mkdtemp(prefix="loa_prompt_")
        prompt_file = Path(temp_dir) / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        cmd = base_cmd + ["-f", str(prompt_file)]

        def _cleanup() -> None:
            try:
                os.unlink(prompt_file)
            except OSError:
                pass
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass

        cleanup = _cleanup

    cmd_with_schema = cmd + [schema_flag, str(schema_path)]

    try:
        try:
            raw = _run_cli(cmd_with_schema, timeout_sec=timeout_sec)
        except Exception:
            raw = _run_cli(cmd, timeout_sec=timeout_sec)
    finally:
        if cleanup is not None:
            cleanup()

    return raw


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
    backend = os.getenv("LOA_LLM_BACKEND", "server").strip().lower()
    timeout_sec = int(os.getenv("LOA_LLM_TIMEOUT_SEC", "90"))

    if backend == "server":
        raw = _run_server(prompt=prompt, schema_path=schema, timeout_sec=timeout_sec, temp=temp, seed=seed)
    elif backend == "cli":
        raw = _run_cli_json(
            prompt=prompt,
            schema_path=schema,
            model_path=model_path,
            n_ctx=n_ctx,
            temp=temp,
            seed=seed,
        )
    else:
        raise LLMError(f"unsupported LOA_LLM_BACKEND: {backend}")

    return _extract_json_object(raw)
