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


def extract_json_object(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise LLMError("model output did not contain a valid JSON object")

    start = text.find("{")
    if start < 0:
        raise LLMError("model output did not contain a valid JSON object")

    in_string = False
    escaped = False
    depth = 0

    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise LLMError("model output did not contain a valid JSON object") from exc
                if not isinstance(parsed, dict):
                    raise LLMError("model output JSON is not an object")
                return parsed

    raise LLMError("model output did not contain a valid JSON object")


def _write_llm_logs(log_dir: Path | None, response_payload, text: str) -> None:
    if log_dir is None:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_path = log_dir / "llm_raw_response.json"
    text_path = log_dir / "llm_text.txt"

    if isinstance(response_payload, (dict, list)):
        raw_path.write_text(json.dumps(response_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        raw_path.write_text(
            json.dumps({"raw": str(response_payload)}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    text_path.write_text(text or "", encoding="utf-8")

    if os.getenv("LOA_DEBUG_LLM", "0") == "1":
        print("\n[LOA_DEBUG_LLM] Extracted LLM text:\n")
        print(text)


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


def _run_server(prompt: str, schema_path: Path, timeout_sec: int, temp: float, seed: int) -> tuple[dict, str]:
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
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"llm server request failed: {exc}") from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMError("llm server response is not valid JSON") from exc

    try:
        choices = data["choices"]
        first = choices[0]
    except Exception as exc:
        raise LLMError("llm server response not in chat completions format") from exc

    content = None
    if isinstance(first, dict):
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
        if not content:
            content = first.get("text")

    if not isinstance(content, str) or not content.strip():
        raise LLMError("llm server returned empty content")

    return data, content


def _run_cli_text(prompt: str, schema_path: Path, model_path: str, n_ctx: int, temp: float, seed: int) -> tuple[dict, str]:
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

    return {"backend": "cli", "stdout": raw}, raw


def run_llm_text(
    prompt: str,
    schema_path: str | Path,
    *,
    model_path: str,
    n_ctx: int,
    temp: float,
    seed: int,
    log_dir: str | Path | None = None,
    timeout_sec_override: int | None = None,
) -> str:
    schema = Path(schema_path).resolve()
    backend = os.getenv("LOA_LLM_BACKEND", "server").strip().lower()
    timeout_sec = timeout_sec_override if timeout_sec_override is not None else int(os.getenv("LOA_LLM_TIMEOUT_SEC", "90"))
    resolved_log_dir = Path(log_dir).resolve() if log_dir else None

    if backend == "server":
        raw_payload, text = _run_server(
            prompt=prompt,
            schema_path=schema,
            timeout_sec=timeout_sec,
            temp=temp,
            seed=seed,
        )
    elif backend == "cli":
        raw_payload, text = _run_cli_text(
            prompt=prompt,
            schema_path=schema,
            model_path=model_path,
            n_ctx=n_ctx,
            temp=temp,
            seed=seed,
        )
    else:
        raise LLMError(f"unsupported LOA_LLM_BACKEND: {backend}")

    _write_llm_logs(resolved_log_dir, raw_payload, text)
    return text


def run_llm_json(
    prompt: str,
    schema_path: str | Path,
    *,
    model_path: str,
    n_ctx: int,
    temp: float,
    seed: int,
    log_dir: str | Path | None = None,
    timeout_sec_override: int | None = None,
) -> dict:
    text = run_llm_text(
        prompt,
        schema_path,
        model_path=model_path,
        n_ctx=n_ctx,
        temp=temp,
        seed=seed,
        log_dir=log_dir,
        timeout_sec_override=timeout_sec_override,
    )
    return extract_json_object(text)
