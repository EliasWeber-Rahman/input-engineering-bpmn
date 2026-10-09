import os
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer, util

from . import config
from .autobpmn_client import get_similarity

model_instance: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global model_instance
    if model_instance is None:
        model_instance = SentenceTransformer(config.SBERT_MODEL_NAME)
    return model_instance


def extract_cpee_labels(xml_path: Path) -> List[str]:
    if not xml_path.exists():
        return []
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        print(f"{xml_path.name} is not valid XML")
        return []
    labels = []
    for element in tree.getroot().iter():
        if element.tag.endswith("label") and element.text and element.text.strip():
            labels.append(element.text.strip())
    return labels


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0


def mean_and_sd(values: List[float]) -> tuple:
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, sd


def calculate_metrics(ground_truth_labels: List[str], generated_labels: List[str],
                      threshold: float = config.MATCH_THRESHOLD) -> dict:
    if not ground_truth_labels or not generated_labels:
        return {
            "precision": 0, "recall": 0, "jaccard": 0,
            "matches": [], "unmatched_gt": ground_truth_labels, "unmatched_gen": generated_labels,
        }

    model = get_model()
    gt_embeddings = model.encode(ground_truth_labels, convert_to_tensor=True)
    gen_embeddings = model.encode(generated_labels, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(gen_embeddings, gt_embeddings).cpu().numpy()

    possible_matches = []
    for generated_index in range(len(generated_labels)):
        for ground_truth_index in range(len(ground_truth_labels)):
            score = scores[generated_index][ground_truth_index]
            if score >= threshold:
                possible_matches.append((score, generated_index, ground_truth_index))
    possible_matches.sort(reverse=True)

    matched_ground_truth_indexes = set()
    used_generated_indexes = set()
    matches = []
    for score, generated_index, ground_truth_index in possible_matches:
        if generated_index not in used_generated_indexes and ground_truth_index not in matched_ground_truth_indexes:
            used_generated_indexes.add(generated_index)
            matched_ground_truth_indexes.add(ground_truth_index)
            matches.append((ground_truth_labels[ground_truth_index], generated_labels[generated_index], float(score)))

    unmatched_gt = []
    for index, label in enumerate(ground_truth_labels):
        if index not in matched_ground_truth_indexes:
            unmatched_gt.append(label)

    unmatched_gen = []
    for index, label in enumerate(generated_labels):
        if index not in used_generated_indexes:
            unmatched_gen.append(label)

    true_positives = len(matches)
    false_positives = len(unmatched_gen)
    false_negatives = len(unmatched_gt)

    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, true_positives + false_negatives)
    jaccard = safe_divide(true_positives, true_positives + false_positives + false_negatives)

    return {
        "precision": round(precision, 4), "recall": round(recall, 4), "jaccard": round(jaccard, 4),
        "matches": matches, "unmatched_gt": unmatched_gt, "unmatched_gen": unmatched_gen,
    }


def ground_truth_path_for_description(dataset: str, description_id: str) -> Optional[Path]:
    gt_dir = config.dataset_ground_truth_dir(dataset)

    if dataset == "domain":
        candidate = gt_dir / f"{description_id}.xml"
        if candidate.exists():
            return candidate
        return None

    if dataset == "pet":
        candidate = gt_dir / f"{description_id.replace('_', '-')}.xml"
        if candidate.exists():
            return candidate
        return None

    if dataset == "realset":
        if not gt_dir.exists():
            return None
        for filename in sorted(os.listdir(gt_dir)):
            if filename.startswith(description_id) and filename.endswith(".xml"):
                return gt_dir / filename
        return None

    raise ValueError(f"Unknown dataset '{dataset}'")


def find_generated_path(gt_filename: str, gen_dir: Path, dataset: str, run_index: int = 1) -> Optional[Path]:
    if run_index == 1:
        suffix = ""
    else:
        suffix = f"_run{run_index}"

    if dataset == "domain":
        candidate = gen_dir / f"{Path(gt_filename).stem}{suffix}.xml"
    elif dataset == "pet":
        candidate = gen_dir / f"{Path(gt_filename).stem.replace('-', '_')}{suffix}.xml"
    elif dataset == "realset":
        prefix = gt_filename[:5]
        if not gen_dir.exists():
            return None
        for filename in os.listdir(gen_dir):
            if filename.startswith(prefix) and filename.endswith(f"{suffix}.xml"):
                return gen_dir / filename
        return None
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    if candidate.exists():
        return candidate
    return None


def evaluate_config(
        dataset: str,
        config_name: str,
        n_runs: int = config.N_RUNS,
        limit: Optional[int] = None,
        model: str = config.AUTOBPMN_DEFAULT_MODEL,
) -> Dict[str, dict]:
    from .dataset_generation import load_descriptions

    gen_dir = config.dataset_generated_dir(dataset, config_name, model=model)
    description_ids = sorted(load_descriptions(dataset, limit=limit).keys())

    results: Dict[str, dict] = {}
    for desc_id in description_ids:
        gt_path = ground_truth_path_for_description(dataset, desc_id)
        if gt_path is None:
            print(f"no ground-truth file found for description '{desc_id}'")
            continue
        gt_labels = extract_cpee_labels(gt_path)

        jaccards = []
        precisions = []
        recalls = []
        trace_similarities = []
        for run_index in range(1, n_runs + 1):
            gen_path = find_generated_path(gt_path.name, gen_dir, dataset, run_index=run_index)
            gen_labels = extract_cpee_labels(gen_path) if gen_path else []
            metrics = calculate_metrics(gt_labels, gen_labels)
            jaccards.append(metrics["jaccard"])
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])

            if gen_path is not None:
                trace_similarity = get_similarity(gt_path, gen_path, sim_type="trace")
                if trace_similarity is not None:
                    trace_similarities.append(trace_similarity)

        jaccard_mean, jaccard_sd = mean_and_sd(jaccards)
        precision_mean, precision_sd = mean_and_sd(precisions)
        recall_mean, recall_sd = mean_and_sd(recalls)
        trace_similarity_mean, trace_similarity_sd = mean_and_sd(trace_similarities)

        results[desc_id] = {
            "jaccard_mean": jaccard_mean,
            "jaccard_sd": jaccard_sd,
            "precision_mean": precision_mean,
            "precision_sd": precision_sd,
            "recall_mean": recall_mean,
            "recall_sd": recall_sd,
            "trace_similarity_mean": trace_similarity_mean,
            "trace_similarity_sd": trace_similarity_sd,
        }
    return results


def paired_delta(
        dataset: str,
        config_name: str,
        baseline_name: str = "baseline",
        n_runs: int = config.N_RUNS,
        model: str = config.AUTOBPMN_DEFAULT_MODEL,
) -> Dict[str, float]:
    config_results = evaluate_config(dataset, config_name, n_runs=n_runs, model=model)
    baseline_results = evaluate_config(dataset, baseline_name, n_runs=n_runs, model=model)

    deltas = {}
    for desc_id in config_results:
        config_jaccard = config_results[desc_id]["jaccard_mean"]
        if desc_id in baseline_results:
            baseline_jaccard = baseline_results[desc_id]["jaccard_mean"]
        else:
            baseline_jaccard = 0.0
        deltas[desc_id] = config_jaccard - baseline_jaccard
    return deltas


def print_summary(
        dataset: str,
        config_names: List[str],
        n_runs: int = config.N_RUNS,
        limit: Optional[int] = None,
        model: str = config.AUTOBPMN_DEFAULT_MODEL,
) -> None:
    for name in config_names:
        results = evaluate_config(dataset, name, n_runs=n_runs, limit=limit, model=model)
        if not results:
            print(f"{dataset}/{name}: no ground-truth files found.")
            continue

        jaccard_means = [result["jaccard_mean"] for result in results.values()]
        jaccard_sds = [result["jaccard_sd"] for result in results.values()]
        precision_means = [result["precision_mean"] for result in results.values()]
        recall_means = [result["recall_mean"] for result in results.values()]
        trace_similarity_means = [result["trace_similarity_mean"] for result in results.values()]

        mean_jaccard = statistics.mean(jaccard_means)
        mean_jaccard_sd = statistics.mean(jaccard_sds)
        mean_precision = statistics.mean(precision_means)
        mean_recall = statistics.mean(recall_means)
        mean_trace_similarity = statistics.mean(trace_similarity_means)
        print(
            f"{model}/{dataset}/{name:20s} Jaccard = {mean_jaccard:.3f} (SD={mean_jaccard_sd:.3f})  "
            f"Precision = {mean_precision:.3f}  Recall = {mean_recall:.3f}  "
            f"Trace sim = {mean_trace_similarity:.3f}  (n={len(results)})"
        )


def find_run_detail(runs: List[dict], run_index: int) -> Optional[dict]:
    for run in runs:
        if run["run_index"] == run_index:
            return run
    return None
