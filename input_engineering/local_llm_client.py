import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests

from . import config
from .autobpmn_client import output_path
FENCE = re.compile(r"```(?:mermaid)?\s*(.*?)```", re.DOTALL)

_system_prompt: Optional[str] = None


class LocalLLMError(RuntimeError):
    pass


def get_system_prompt() -> str:
    global _system_prompt
    if _system_prompt is None:
        _system_prompt = config.LOCAL_LLM_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return _system_prompt


def generate_mermaid(
    description: str,
    model: str = config.LOCAL_LLM_MODEL,
    temperature: float = config.LOCAL_LLM_TEMPERATURE,
    max_tokens: int = config.LOCAL_LLM_MAX_TOKENS,
) -> str:
    payload = {
        "model": model,
        "prompt": config.LOCAL_LLM_USER_PROMPT_TEMPLATE.format(description=description),
        "system": get_system_prompt(),
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        response = requests.post(config.OLLAMA_URL, json=payload, timeout=config.OLLAMA_TIMEOUT_S)
    except requests.exceptions.ConnectionError as exc:
        raise LocalLLMError(
            f"Could not reach Ollama at {config.OLLAMA_URL}"
        ) from exc

    if response.status_code != 200:
        raise LocalLLMError(f"Ollama HTTP {response.status_code}: {response.text[:500]}")

    result = json.loads(response.text)
    raw_text = result.get("response", "").strip()
    if not raw_text:
        raise LocalLLMError(f"Ollama returned an empty response: {response.text[:500]}")
    return extract_mermaid(raw_text)


def extract_mermaid(text: str) -> str:
    fence_match = FENCE.search(text)
    if fence_match:
        text = fence_match.group(1)
    header_index = text.find("graph LR")
    if header_index > 0:
        text = text[header_index:]
    return text.strip()


def is_plausible_mermaid(text: str) -> bool:
    return "graph LR" in text and "-->" in text


def mermaid_output_path(
    dataset: str,
    config_name: str,
    description_id: str,
    run_index: int = 1,
    model: str = config.LOCAL_LLM_MODEL,
) -> Path:
    directory = config.mermaid_output_dir(dataset, config_name, model=model)
    suffix = "" if run_index == 1 else f"_run{run_index}"
    return directory / f"{description_id}{suffix}.mmd"


def generate_mermaid_and_save(
    description: str,
    dataset: str,
    config_name: str,
    description_id: str,
    run_index: int = 1,
    model: str = config.LOCAL_LLM_MODEL,
) -> Path:
    mermaid_text = generate_mermaid(description, model=model)
    path = mermaid_output_path(dataset, config_name, description_id, run_index, model=model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mermaid_text, encoding="utf-8")
    return path


def _extract_cpee_xml(response_text: str) -> Optional[str]:
    stripped = response_text.strip()
    if not stripped:
        return None
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("output_cpee", "cpee", "xml", "description", "result"):
                if payload.get(key):
                    return payload[key]
            return None
    if stripped.startswith("<"):
        return stripped
    return None


def convert_mermaid_to_cpee(mermaid_text: str, max_retries: int = config.AUTOBPMN_MAX_RETRIES) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".mmd", delete=False, encoding="utf-8"
            ) as tmp_file:
                tmp_file.write(mermaid_text)
                tmp_path = tmp_file.name

            result = subprocess.run(
                [
                    "curl", "-sS", "-X", "POST", config.CPEE_MERMAID_CONVERT_URL,
                    "-F", f"description=@{tmp_path};type=text/plain",
                    "-F", "type=description",
                ],
                capture_output=True,
                text=True,
                timeout=config.AUTOBPMN_TIMEOUT_S,
            )

            if result.returncode != 0:
                last_error = LocalLLMError(
                    f"curl exited {result.returncode}: {result.stderr[:500]!r}"
                )
            else:
                cpee_xml = _extract_cpee_xml(result.stdout)
                if cpee_xml:
                    return cpee_xml
                last_error = LocalLLMError(
                    f"curl succeeded but couldn't find CPEE XML in the output: {result.stdout[:500]!r}"
                )

        except FileNotFoundError as exc:
            raise LocalLLMError(
                "`curl` was not found on PATH "
            ) from exc
        except Exception as exc:
            last_error = exc
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)

        if attempt < max_retries:
            time.sleep(2 ** attempt)

    raise LocalLLMError(f"Mermaid-to-CPEE conversion failed after {max_retries} attempts: {last_error}") from last_error


def convert_and_save(
    mermaid_path: Path,
    dataset: str,
    config_name: str,
    description_id: str,
    run_index: int = 1,
    model: str = config.LOCAL_LLM_MODEL,
) -> Path:
    mermaid_text = mermaid_path.read_text(encoding="utf-8")
    cpee_xml = convert_mermaid_to_cpee(mermaid_text)
    path = output_path(dataset, config_name, description_id, run_index, model=model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cpee_xml, encoding="utf-8")
    return path
