import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESCRIPTIONS_DIR = PROJECT_ROOT / "descriptions"
CPEE_MODELS_DIR = PROJECT_ROOT / "cpee-models"
GROUND_TRUTH_DIR = CPEE_MODELS_DIR / "ground_truth"
GENERATED_DIR = CPEE_MODELS_DIR / "generated"
GLOSSARY_PATH = Path(__file__).resolve().parent / "glossary.json"
RESULTS_DIR = PROJECT_ROOT / "results"
DATASETS = ["pet", "domain", "realset"]
INJECTABLE_DATASETS = ["pet"]
AUTOBPMN_URL = "https://autobpmn.ai/llm/"
AUTOBPMN_DEFAULT_MODEL = "gemini-3.1-flash-lite"
AUTOBPMN_TIMEOUT_S = 60
AUTOBPMN_MAX_RETRIES = 3
CPEE_SIMILARITY_URL = "https://cpee.org/similarity/"
CPEE_SIMILARITY_DEFAULT_THRESHOLD = 0.7

def model_choice(model: str) -> str:
    if model == LOCAL_LLM_MODEL:
        return "llama3.1-8b-local"
    return model


AUTOBPMN_PROMPT_TYPE_GENERATE = "generate_endpoints"
N_RUNS = 5
NOISE_TARGET_EDIT_RATE = 0.15
PET_INJECTED_DIR = DESCRIPTIONS_DIR / "pet_injected"
PET_INJECTED_N_RUNS = 3
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
MATCH_THRESHOLD = 0.6
SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")

COMPONENT_ORDER: List[str] = [
    "normalization",
    "abbreviation",
    "gec",
    "coreference",
    "sentence_decomposition",
]


def all_components_enabled_by_default() -> Dict[str, bool]:
    enabled = {}
    for component_name in COMPONENT_ORDER:
        enabled[component_name] = True
    return enabled


@dataclass
class PipelineConfig:
    name: str
    enabled: Dict[str, bool] = field(default_factory=all_components_enabled_by_default)

    def is_enabled(self, component: str) -> bool:
        return self.enabled.get(component, False)


def dataset_descriptions_dir(dataset: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {DATASETS}.")
    return DESCRIPTIONS_DIR / dataset


def dataset_ground_truth_dir(dataset: str) -> Path:
    return GROUND_TRUTH_DIR / dataset / "cpee_xml"


def dataset_generated_dir(dataset: str, config_name: Optional[str] = None, model: str = AUTOBPMN_DEFAULT_MODEL) -> Path:
    base = GENERATED_DIR / model_choice(model) / dataset
    if config_name is None or config_name == "baseline":
        return base
    return base / config_name


def results_dir(model: str = AUTOBPMN_DEFAULT_MODEL) -> Path:
    return RESULTS_DIR / model_choice(model)

LOCAL_LLM_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT_S = 300
LOCAL_LLM_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "local_llm_system_prompt.txt"
LOCAL_LLM_USER_PROMPT_TEMPLATE = "Consider following process description: {description}. Generate a BPMN model in Mermaid.js format."
LOCAL_LLM_TEMPERATURE = 0
LOCAL_LLM_MAX_TOKENS = 4000
CPEE_MERMAID_CONVERT_URL = "https://cpee.org/transformation/cpee/mermaid/"
MERMAID_DIR = CPEE_MODELS_DIR / "mermaid"


def mermaid_output_dir(dataset: str, config_name: Optional[str] = None, model: str = LOCAL_LLM_MODEL) -> Path:
    base = MERMAID_DIR / model_choice(model) / dataset
    if config_name is None or config_name == "baseline":
        return base
    return base / config_name
