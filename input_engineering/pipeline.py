from typing import Dict, List

from . import config
from .components.normalization import normalize
from .components.abbreviation import expand_abbreviations
from .components.gec import correct_grammar
from .components.coreference import resolve_coreferences
from .components.sentence_decomposition import decompose_sentences
from .components.base import ComponentResult


class Pipeline:
    def __init__(self, pipeline_config: config.PipelineConfig):
        self.config = pipeline_config

    def run(self, text: str) -> Dict[str, object]:
        current_text = text
        trace: List[ComponentResult] = []

        for component_name in config.COMPONENT_ORDER:
            if not self.config.is_enabled(component_name):
                continue
            result = self.run_component(component_name, current_text)
            trace.append(result)
            current_text = result.text

        return {"text": current_text, "trace": trace, "config_name": self.config.name}

    def run_component(self, component_name: str, text: str) -> ComponentResult:
        if component_name == "normalization":
            return normalize(text)
        if component_name == "abbreviation":
            return expand_abbreviations(text)
        if component_name == "gec":
            return correct_grammar(text)
        if component_name == "coreference":
            return resolve_coreferences(text)
        if component_name == "sentence_decomposition":
            return decompose_sentences(text)
        raise ValueError(f"Unknown component '{component_name}'.")


def build_enabled_flags(turn_all_on: bool) -> Dict[str, bool]:
    enabled_flags = {}
    for component_name in config.COMPONENT_ORDER:
        enabled_flags[component_name] = turn_all_on
    return enabled_flags


def complete() -> Dict[str, config.PipelineConfig]:
    configs: Dict[str, config.PipelineConfig] = {}

    configs["baseline"] = config.PipelineConfig(name="baseline", enabled=build_enabled_flags(False))
    configs["full"] = config.PipelineConfig(name="full", enabled=build_enabled_flags(True))

    for component_name in config.COMPONENT_ORDER:
        isolated_flags = build_enabled_flags(False)
        isolated_flags[component_name] = True
        configs[component_name] = config.PipelineConfig(name=component_name, enabled=isolated_flags)

    for component_name in config.COMPONENT_ORDER:
        full_except_flags = build_enabled_flags(True)
        full_except_flags[component_name] = False
        config_name = f"full_except_{component_name}"
        configs[config_name] = config.PipelineConfig(name=config_name, enabled=full_except_flags)

    return configs
