import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import config
from .autobpmn_client import generate_and_save, output_path
from .inject_noise import NOISE_INJECTORS, inject_combined_noise
from .noise_metrics import edit_rate
from .pipeline import Pipeline, complete


def report_noise_calibration(dataset: str = "pet", target_edit_rate: Optional[float] = None, seed: int = 42) -> None:
    from .components.normalization import normalize

    if target_edit_rate is None:
        target_edit_rate = config.NOISE_TARGET_EDIT_RATE
    rng = random.Random(seed)
    descriptions = load_descriptions(dataset)

    achieved_rates: Dict[str, List[float]] = {name: [] for name in NOISE_INJECTORS}
    below_target_count: Dict[str, int] = {name: 0 for name in NOISE_INJECTORS}

    for raw_text in descriptions.values():
        clean_text = normalize(raw_text).text
        for component, inject in NOISE_INJECTORS.items():
            noised = inject(clean_text, rng, target_edit_rate)
            rate = edit_rate(clean_text, noised)
            achieved_rates[component].append(rate)
            if rate < target_edit_rate:
                below_target_count[component] += 1

    print(f"target edit_rate = {target_edit_rate:.3f} (edits/token), n = {len(descriptions)} descriptions\n")
    header = f"{'component':<24}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}{'below target':>14}"
    print(header)
    print("-" * len(header))
    for component, rates in achieved_rates.items():
        mean = statistics.mean(rates) if rates else 0.0
        sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
        print(f"{component:<24}{mean:>8.3f}{sd:>8.3f}{min(rates, default=0.0):>8.3f}"
              f"{max(rates, default=0.0):>8.3f}{below_target_count[component]:>14}")


def load_descriptions(dataset: str, limit: Optional[int] = None) -> Dict[str, str]:
    directory = config.dataset_descriptions_dir(dataset)
    paths = sorted(directory.glob("*.txt"))
    if limit is not None:
        paths = paths[:limit]
    return {path.stem: path.read_text(encoding="utf-8") for path in paths}


def build_injected_pet_set(
        seed: int = 42, target_edit_rate: Optional[float] = None, record_edit_rates: bool = False
) -> Dict[str, Dict[str, str]]:
    from .components.normalization import normalize

    if target_edit_rate is None:
        target_edit_rate = config.NOISE_TARGET_EDIT_RATE
    rng = random.Random(seed)
    descriptions = load_descriptions("pet")

    result: Dict[str, Dict[str, str]] = {}
    for desc_id, raw_text in descriptions.items():
        clean_text = normalize(raw_text).text
        result[desc_id] = {"clean": clean_text}
        for component, inject in NOISE_INJECTORS.items():
            noised = inject(clean_text, rng, target_edit_rate)
            result[desc_id][component] = noised
            if record_edit_rates:
                result[desc_id][f"{component}__edit_rate"] = edit_rate(clean_text, noised)

        combined = inject_combined_noise(clean_text, rng, target_edit_rate)
        result[desc_id]["combined"] = combined
        if record_edit_rates:
            result[desc_id]["combined__edit_rate"] = edit_rate(clean_text, combined)
    return result


def save_injected_pet_set(injected: Dict[str, Dict[str, str]], output_dir: Optional[Path] = None) -> None:
    output_dir = config.PET_INJECTED_DIR if output_dir is None else output_dir

    written = 0
    no_change = 0
    for desc_id, variants in injected.items():
        clean_text = variants["clean"]
        for category, text in variants.items():
            if category.endswith("__edit_rate"):
                continue
            if category != "clean" and edit_rate(clean_text, text) <= 1e-9:
                no_change += 1
                continue
            category_dir = output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            (category_dir / f"{desc_id}.txt").write_text(text, encoding="utf-8")
            written += 1
    print(f"Wrote {written} files under {output_dir}, {no_change} with no actual change")


def generate_dataset_config(
        dataset: str,
        config_name: str,
        pipeline_config,
        descriptions: Dict[str, str],
        n_runs: int = config.N_RUNS,
        model: str = config.AUTOBPMN_DEFAULT_MODEL,
) -> None:
    pipeline = Pipeline(pipeline_config)
    for desc_id, text in descriptions.items():
        preprocessed = pipeline.run(text)["text"]
        for run_index in range(1, n_runs + 1):
            try:
                path = generate_and_save(preprocessed, dataset, config_name, desc_id, run_index=run_index, model=model)
                print(f"[{model}/{dataset}/{config_name}] {desc_id} run {run_index} -> {path}")
            except Exception as exc:
                print(f"[{model}/{dataset}/{config_name}] {desc_id} run {run_index} FAILED: {exc}")


def generate_all(dataset: str, items: List[Tuple[str, str, str]], n_runs: int,
                 model: str = config.AUTOBPMN_DEFAULT_MODEL) -> None:
    total = skipped = failed = 0
    for config_name, desc_id, text in items:
        for run_index in range(1, n_runs + 1):
            total += 1
            run_config_name = f"{config_name}/run_{run_index}"
            if output_path(dataset, run_config_name, desc_id, model=model).exists():
                skipped += 1
                continue
            try:
                path = generate_and_save(text, dataset, run_config_name, desc_id, model=model)
                print(f"[{model}/{dataset}/{run_config_name}] {desc_id} -> {path}")
            except Exception as exc:
                failed += 1
                print(f"[{model}/{dataset}/{run_config_name}] {desc_id} FAILED: {exc}")

    print(f"\nDone: {total - skipped - failed}/{total} succeeded, {skipped} already done (skipped), {failed} failed.")


def generate_pet_injected(
        n_runs: int = config.PET_INJECTED_N_RUNS,
        limit: Optional[int] = None,
        categories: Optional[List[str]] = None,
        model: str = config.AUTOBPMN_DEFAULT_MODEL,
) -> None:
    category_dirs = sorted(p for p in config.PET_INJECTED_DIR.iterdir() if p.is_dir())
    if categories is not None:
        category_dirs = [p for p in category_dirs if p.name in categories]

    items: List[Tuple[str, str, str]] = []
    for category_dir in category_dirs:
        text_paths = sorted(category_dir.glob("*.txt"))
        if limit is not None:
            text_paths = text_paths[:limit]
        for text_path in text_paths:
            text = text_path.read_text(encoding="utf-8")
            items.append((category_dir.name, text_path.stem, text))

    generate_all("pet_injected", items, n_runs, model=model)


RQ3_FULL_CATEGORIES = ["combined", "coreference"]
RQ3_LIGHT_CATEGORIES = ["normalization", "gec", "sentence_decomposition", "abbreviation"]


def default_category_configs() -> Dict[str, List[str]]:
    config_names = list(complete())
    category_configs: Dict[str, List[str]] = {}
    for category in RQ3_FULL_CATEGORIES:
        category_configs[category] = config_names
    for category in RQ3_LIGHT_CATEGORIES:
        category_configs[category] = ["baseline", "full"]
    return category_configs


DETERMINISTIC_PIPELINE_DESCRIPTIONS_DIR = config.DESCRIPTIONS_DIR / "deterministic_pipeline" / "pet_injected"


def build_pipeline_injected(
        category_configs: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    if category_configs is None:
        category_configs = default_category_configs()
    config_type = complete()

    result: Dict[str, Dict[str, Dict[str, str]]] = {}
    for category, config_names in category_configs.items():
        category_dir = config.PET_INJECTED_DIR / category
        text_paths = sorted(category_dir.glob("*.txt"))

        result[category] = {}
        for config_name in config_names:
            if config_name not in config_type:
                raise ValueError(f"Unknown config '{config_name}'")
            pipeline = Pipeline(config_type[config_name])

            preprocessed_by_id: Dict[str, str] = {}
            for text_path in text_paths:
                desc_id = text_path.stem
                raw_text = text_path.read_text(encoding="utf-8")
                preprocessed_by_id[desc_id] = pipeline.run(raw_text)["text"]
            result[category][config_name] = preprocessed_by_id
    return result


def save_pipeline_injected(
        preprocessed: Dict[str, Dict[str, Dict[str, str]]],
        output_dir: Optional[Path] = None,
) -> None:
    output_dir = DETERMINISTIC_PIPELINE_DESCRIPTIONS_DIR if output_dir is None else output_dir

    written = 0
    for category, by_config in preprocessed.items():
        for config_name, by_desc_id in by_config.items():
            config_dir = output_dir / category / config_name
            config_dir.mkdir(parents=True, exist_ok=True)
            for desc_id, text in by_desc_id.items():
                (config_dir / f"{desc_id}.txt").write_text(text, encoding="utf-8")
                written += 1
    print(f"Wrote {written} files under {output_dir}")


def generate_pipeline_injected(
        n_runs: int = config.PET_INJECTED_N_RUNS,
        limit: Optional[int] = None,
        base_dir: Optional[Path] = None,
        categories: Optional[List[str]] = None,
        model: str = config.AUTOBPMN_DEFAULT_MODEL,
) -> None:
    base_dir = DETERMINISTIC_PIPELINE_DESCRIPTIONS_DIR if base_dir is None else base_dir
    category_dirs = sorted(p for p in base_dir.iterdir() if p.is_dir())
    if categories is not None:
        category_dirs = [p for p in category_dirs if p.name in categories]

    items: List[Tuple[str, str, str]] = []
    for category_dir in category_dirs:
        config_dirs = sorted(p for p in category_dir.iterdir() if p.is_dir())
        for config_dir in config_dirs:
            text_paths = sorted(config_dir.glob("*.txt"))
            if limit is not None:
                text_paths = text_paths[:limit]
            for text_path in text_paths:
                text = text_path.read_text(encoding="utf-8")
                generation_config_name = f"pet_injected/{category_dir.name}/{config_dir.name}"
                items.append((generation_config_name, text_path.stem, text))

    generate_all("deterministic_pipeline", items, n_runs, model=model)


def generate_mermaid_local(
        dataset: str,
        items: List[Tuple[str, str, str]],
        n_runs: int = 1,
        model: str = config.LOCAL_LLM_MODEL,
) -> Dict[str, int]:
    from .local_llm_client import generate_mermaid_and_save, mermaid_output_path, is_plausible_mermaid

    total = skipped = failed = invalid = 0
    valid_by_config: Dict[str, int] = {}
    total_by_config: Dict[str, int] = {}

    for config_name, desc_id, text in items:
        for run_index in range(1, n_runs + 1):
            total += 1
            total_by_config[config_name] = total_by_config.get(config_name, 0) + 1
            run_config_name = f"{config_name}/run_{run_index}"

            existing = mermaid_output_path(dataset, run_config_name, desc_id, model=model)
            if existing.exists():
                skipped += 1
                if is_plausible_mermaid(existing.read_text(encoding="utf-8")):
                    valid_by_config[config_name] = valid_by_config.get(config_name, 0) + 1
                continue

            try:
                path = generate_mermaid_and_save(text, dataset, run_config_name, desc_id, model=model)
                if is_plausible_mermaid(path.read_text(encoding="utf-8")):
                    valid_by_config[config_name] = valid_by_config.get(config_name, 0) + 1
                else:
                    invalid += 1
                print(f"[{model}/{dataset}/{run_config_name}] {desc_id} -> {path}")
            except Exception as exc:
                failed += 1
                print(f"[{model}/{dataset}/{run_config_name}] {desc_id} FAILED: {exc}")

    print(
        f"\nDone: {total - skipped - failed}/{total} generated "
        f"({skipped} already done, {failed} failed outright, {invalid} produced but not diagram-shaped)."
    )
    for config_name, total_count in total_by_config.items():
        valid_count = valid_by_config.get(config_name, 0)
        print(f"  {config_name:20s} {valid_count}/{total_count} plausible Mermaid diagrams")

    return valid_by_config


def generate_mermaid_pet_injected(
        limit: Optional[int] = None,
        categories: Optional[List[str]] = None,
        n_runs: int = 1,
        model: str = config.LOCAL_LLM_MODEL,
) -> None:
    category_dirs = sorted(p for p in config.PET_INJECTED_DIR.iterdir() if p.is_dir())
    if categories is not None:
        category_dirs = [p for p in category_dirs if p.name in categories]

    items: List[Tuple[str, str, str]] = []
    for category_dir in category_dirs:
        text_paths = sorted(category_dir.glob("*.txt"))
        if limit is not None:
            text_paths = text_paths[:limit]
        for text_path in text_paths:
            text = text_path.read_text(encoding="utf-8")
            items.append((category_dir.name, text_path.stem, text))

    generate_mermaid_local("pet_injected", items, n_runs=n_runs, model=model)


def generate_mermaid_pipeline_injected(
        limit: Optional[int] = None,
        categories: Optional[List[str]] = None,
        base_dir: Optional[Path] = None,
        n_runs: int = 1,
        model: str = config.LOCAL_LLM_MODEL,
) -> None:
    base_dir = DETERMINISTIC_PIPELINE_DESCRIPTIONS_DIR if base_dir is None else base_dir
    category_dirs = sorted(p for p in base_dir.iterdir() if p.is_dir())
    if categories is not None:
        category_dirs = [p for p in category_dirs if p.name in categories]

    items: List[Tuple[str, str, str]] = []
    for category_dir in category_dirs:
        config_dirs = sorted(p for p in category_dir.iterdir() if p.is_dir())
        for config_dir in config_dirs:
            text_paths = sorted(config_dir.glob("*.txt"))
            if limit is not None:
                text_paths = text_paths[:limit]
            for text_path in text_paths:
                text = text_path.read_text(encoding="utf-8")
                generation_config_name = f"pet_injected/{category_dir.name}/{config_dir.name}"
                items.append((generation_config_name, text_path.stem, text))

    generate_mermaid_local("deterministic_pipeline", items, n_runs=n_runs, model=model)


def generate_mermaid_baseline_vs_full(
        limit: Optional[int] = None,
        n_runs: int = 1,
        model: str = config.LOCAL_LLM_MODEL,
) -> None:
    descriptions = load_descriptions("pet", limit=limit)
    suite = complete()

    items: List[Tuple[str, str, str]] = []
    for config_name in ["baseline", "full"]:
        pipeline = Pipeline(suite[config_name])
        for desc_id, text in descriptions.items():
            preprocessed = pipeline.run(text)["text"]
            items.append((config_name, desc_id, preprocessed))

    generate_mermaid_local("pet", items, n_runs=n_runs, model=model)


def convert_mermaid_dataset(dataset: str, model: str = config.LOCAL_LLM_MODEL, limit: Optional[int] = None) -> None:
    from .local_llm_client import convert_and_save

    root = config.mermaid_output_dir(dataset, config_name=None, model=model)
    mmd_paths = sorted(root.rglob("*.mmd"))
    if limit is not None:
        mmd_paths = mmd_paths[:limit]

    total = skipped = failed = 0
    for mmd_path in mmd_paths:
        total += 1
        relative = mmd_path.relative_to(root)
        run_config_name = "/".join(relative.parts[:-1])
        desc_id = relative.stem

        if output_path(dataset, run_config_name, desc_id, model=model).exists():
            skipped += 1
            continue

        try:
            path = convert_and_save(mmd_path, dataset, run_config_name, desc_id, model=model)
            print(f"[{model}/{dataset}/{run_config_name}] {desc_id} -> {path}")
        except Exception as exc:
            failed += 1
            print(f"[{model}/{dataset}/{run_config_name}] {desc_id} FAILED: {exc}")

    print(f"\nDone: {total - skipped - failed}/{total} converted, {skipped} already done (skipped), {failed} failed.")
