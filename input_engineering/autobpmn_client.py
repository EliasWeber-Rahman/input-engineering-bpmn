import json
import time
from pathlib import Path
from typing import Optional

import requests

from . import config

RPST_XML = '<description xmlns="http://cpee.org/ns/description/1.0"/>'

class AutoBPMNError(RuntimeError):
    pass

def generate_model(
    description: str,
    model: str = config.AUTOBPMN_DEFAULT_MODEL,
    max_retries: int = config.AUTOBPMN_MAX_RETRIES,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            files = {
                "rpst_xml": ("cpee_empty_example", RPST_XML, "text/xml"),
                "user_input": (None, description, "text/plain"),
                "llm": (None, model, "text/plain"),
                "prompt_type": (None, config.AUTOBPMN_PROMPT_TYPE_GENERATE, "text/plain"),
            }

            response = requests.post(config.AUTOBPMN_URL, files=files, timeout=config.AUTOBPMN_TIMEOUT_S)

            if response.status_code == 200:
                payload = json.loads(response.text)
                output_cpee = payload.get("output_cpee")
                if output_cpee:
                    return output_cpee
                last_error = AutoBPMNError(
                    f"HTTP 200 but 'output_cpee' was empty/missing: {response.text[:500]}"
                )
            else:
                last_error = AutoBPMNError(f"HTTP {response.status_code}: {response.text[:500]}")

        except Exception as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(2 ** attempt)

    raise AutoBPMNError(f"AutoBPMN generation failed after {max_retries} attempts: {last_error}") from last_error


def output_path(
    dataset: str,
    config_name: str,
    description_id: str,
    run_index: int = 1,
    model: str = config.AUTOBPMN_DEFAULT_MODEL,
) -> Path:
    directory = config.dataset_generated_dir(dataset, config_name, model=model)
    if run_index == 1:
        suffix = ""
    else:
        suffix = f"_run{run_index}"
    return directory / f"{description_id}{suffix}.xml"


def generate_and_save(
    description: str,
    dataset: str,
    config_name: str,
    description_id: str,
    run_index: int = 1,
    model: str = config.AUTOBPMN_DEFAULT_MODEL,
) -> Path:
    output_cpee = generate_model(description, model=model)
    path = output_path(dataset, config_name, description_id, run_index, model=model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output_cpee, encoding="utf-8")
    return path


def get_similarity(
    xml_path_a: Path,
    xml_path_b: Path,
    sim_type: str = "trace",
    threshold: Optional[float] = None,
    max_retries: int = config.AUTOBPMN_MAX_RETRIES,
) -> Optional[float]:
    if not xml_path_a.exists() or not xml_path_b.exists():
        return None

    xml_a = xml_path_a.read_text(encoding="utf-8")
    xml_b = xml_path_b.read_text(encoding="utf-8")

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            files = {
                "rpst_xml_a": (xml_path_a.name, xml_a, "text/xml"),
                "rpst_xml_b": (xml_path_b.name, xml_b, "text/xml"),
                "sim_type": (None, sim_type, "text/plain"),
            }
            if threshold is not None:
                files["threshold"] = (None, str(threshold), "text/plain")

            response = requests.get(config.CPEE_SIMILARITY_URL, files=files, timeout=config.AUTOBPMN_TIMEOUT_S)

            if response.status_code == 200:
                payload = json.loads(response.text)
                similarity = payload.get("similarity")
                if similarity is not None:
                    return float(similarity)
                last_error = AutoBPMNError(
                    f"HTTP 200 but 'similarity' was empty/missing: {response.text[:500]}"
                )
            else:
                last_error = AutoBPMNError(f"HTTP {response.status_code}: {response.text[:500]}")

        except Exception as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(2 ** attempt)

    print(f"trace similarity request failed after {max_retries} attempts ({xml_path_a.name} vs {xml_path_b.name}): {last_error}")
    return None
