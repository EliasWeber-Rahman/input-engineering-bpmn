import statistics
from pathlib import Path
from typing import Dict, List, Optional

from . import config
from .autobpmn_client import get_similarity
from .run_evaluation import calculate_metrics, extract_cpee_labels, find_run_detail, ground_truth_path_for_description


def evaluate_pet_injected(
        category: str,
        n_runs: int = config.PET_INJECTED_N_RUNS,
        model: str = config.AUTOBPMN_DEFAULT_MODEL,
) -> Dict[str, dict]:
    description_dir = config.PET_INJECTED_DIR / category
    category_generated_dir = config.dataset_generated_dir("pet_injected", category, model=model)
    description_ids = sorted(p.stem for p in description_dir.glob("*.txt"))

    results: Dict[str, dict] = {}
    for desc_id in description_ids:
        gt_path = ground_truth_path_for_description("pet", desc_id)
        if gt_path is None:
            print(f"no ground-truth file found for description '{desc_id}'")
            continue
        gt_labels = extract_cpee_labels(gt_path)

        jaccards = []
        precisions = []
        recalls = []
        trace_similarities = []
        run_details = []
        failed_runs = 0
        for run_index in range(1, n_runs + 1):
            gen_path = category_generated_dir / f"run_{run_index}" / f"{desc_id}.xml"
            generation_failed = not gen_path.exists()
            if generation_failed:
                failed_runs += 1
            gen_labels = extract_cpee_labels(gen_path) if gen_path.exists() else []

            metrics = calculate_metrics(gt_labels, gen_labels)
            jaccards.append(metrics["jaccard"])
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])

            trace_similarity = None if generation_failed else get_similarity(gt_path, gen_path, sim_type="trace")
            if trace_similarity is not None:
                trace_similarities.append(trace_similarity)

            run_details.append({
                "run_index": run_index,
                "generation_failed": generation_failed,
                "matches": metrics["matches"],
                "unmatched_gt": metrics["unmatched_gt"],
                "unmatched_gen": metrics["unmatched_gen"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "jaccard": metrics["jaccard"],
                "trace_similarity": trace_similarity,
            })

        trace_similarity_mean = statistics.mean(trace_similarities) if trace_similarities else 0.0
        trace_similarity_sd = statistics.pstdev(trace_similarities) if len(trace_similarities) > 1 else 0.0

        results[desc_id] = {
            "n_runs": n_runs,
            "failed_runs": failed_runs,
            "jaccard_mean": statistics.mean(jaccards),
            "jaccard_sd": statistics.pstdev(jaccards) if len(jaccards) > 1 else 0.0,
            "precision_mean": statistics.mean(precisions),
            "recall_mean": statistics.mean(recalls),
            "trace_similarity_mean": trace_similarity_mean,
            "trace_similarity_sd": trace_similarity_sd,
            "runs": run_details,
        }
    return results


def paired_test_against_clean(
        category: str,
        n_runs: int = config.PET_INJECTED_N_RUNS,
        model: str = config.AUTOBPMN_DEFAULT_MODEL,
        metric: str = "jaccard_mean",
) -> Optional[dict]:
    if category == "clean":
        return None

    from scipy.stats import wilcoxon

    clean_results = evaluate_pet_injected("clean", n_runs=n_runs, model=model)
    category_results = evaluate_pet_injected(category, n_runs=n_runs, model=model)

    shared_ids = sorted(set(clean_results) & set(category_results))
    deltas = [category_results[d][metric] - clean_results[d][metric] for d in shared_ids]

    if len(deltas) < 2 or all(d == 0 for d in deltas):
        return None

    statistic, p_value = wilcoxon(deltas)
    return {
        "n_pairs": len(deltas),
        "mean_delta": statistics.mean(deltas),
        "median_delta": statistics.median(deltas),
        "n_worse": sum(1 for d in deltas if d < 0),
        "n_better": sum(1 for d in deltas if d > 0),
        "n_tied": sum(1 for d in deltas if d == 0),
        "statistic": float(statistic),
        "p_value": float(p_value),
    }


def write_pet_injected_report(
        category: str,
        output_dir: Optional[Path] = None,
        n_runs: int = config.PET_INJECTED_N_RUNS,
        detail_run_index: int = 1,
        model: str = config.AUTOBPMN_DEFAULT_MODEL,
) -> Path:
    output_dir = (config.results_dir(model) / "pet_injected") if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_pet_injected(category, n_runs=n_runs, model=model)
    n = len(results)
    total_runs = n * n_runs
    total_failed_runs = sum(r["failed_runs"] for r in results.values())
    descriptions_with_a_failure = sum(1 for r in results.values() if r["failed_runs"] > 0)

    jaccard_means = [r["jaccard_mean"] for r in results.values()]
    precision_means = [r["precision_mean"] for r in results.values()]
    recall_means = [r["recall_mean"] for r in results.values()]
    trace_similarity_means = [r["trace_similarity_mean"] for r in results.values()]
    mean_jaccard = statistics.mean(jaccard_means) if jaccard_means else 0.0
    sd_jaccard = statistics.pstdev(jaccard_means) if len(jaccard_means) > 1 else 0.0
    mean_precision = statistics.mean(precision_means) if precision_means else 0.0
    mean_recall = statistics.mean(recall_means) if recall_means else 0.0
    mean_trace_similarity = statistics.mean(trace_similarity_means) if trace_similarity_means else 0.0

    lines: List[str] = []
    lines.append(f"# pet_injected/{category} -- evaluation against PET ground truth")
    lines.append("")
    lines.append(
        f"model = `{model}`, n = {n} descriptions x {n_runs} runs each "
        f"({total_failed_runs} of {total_runs} runs failed generation, across {descriptions_with_a_failure} "
        f"descriptions)"
    )
    lines.append("")
    lines.append("## Per description (mean over runs)")
    lines.append("")
    lines.append("| desc_id | jaccard | precision | recall | trace similarity | failed runs |")
    lines.append("|---|---|---|---|---|---|")
    for desc_id, r in results.items():
        lines.append(
            f"| {desc_id} | {r['jaccard_mean']:.3f} (±{r['jaccard_sd']:.3f}) | {r['precision_mean']:.3f} | "
            f"{r['recall_mean']:.3f} | {r['trace_similarity_mean']:.3f} (±{r['trace_similarity_sd']:.3f}) | "
            f"{r['failed_runs']}/{r['n_runs']} |"
        )
    lines.append("")

    lines.append(f"## Label detail per description (run {detail_run_index} of {n_runs}, as an example)")
    lines.append("")
    for desc_id, r in results.items():
        lines.append(f"### {desc_id}")
        run = find_run_detail(r["runs"], detail_run_index)
        if run is None or run["generation_failed"]:
            lines.append("")
            lines.append(f"(no generated XML for run {detail_run_index} -- AutoBPMN generation failed)")
            lines.append("")
            continue
        if run["trace_similarity"] is not None:
            lines.append("")
            lines.append(f"trace similarity = {run['trace_similarity']:.3f}")
        if run["matches"]:
            lines.append("")
            lines.append("MATCHED:")
            for gt_lbl, gen_lbl, score in run["matches"]:
                lines.append(f"- [{score:.2f}] GT `{gt_lbl}` <-> GEN `{gen_lbl}`")
        if run["unmatched_gt"]:
            lines.append("")
            lines.append("MISSED (in ground truth, not generated -- false negatives):")
            for lbl in run["unmatched_gt"]:
                lines.append(f"- {lbl}")
        if run["unmatched_gen"]:
            lines.append("")
            lines.append("EXTRA (generated, not in ground truth -- false positives):")
            for lbl in run["unmatched_gen"]:
                lines.append(f"- {lbl}")
        lines.append("")

    lines.append("## Category summary (whole dataset, this category)")
    lines.append("")
    lines.append(f"- n = {n} descriptions x {n_runs} runs each")
    lines.append(
        f"- {total_failed_runs} of {total_runs} runs failed generation, across {descriptions_with_a_failure} descriptions")
    lines.append(f"- mean Jaccard = {mean_jaccard:.3f} (SD={sd_jaccard:.3f})")
    lines.append(f"- mean Precision = {mean_precision:.3f}")
    lines.append(f"- mean Recall = {mean_recall:.3f}")
    lines.append(f"- mean trace similarity = {mean_trace_similarity:.3f}")
    lines.append("")

    test = paired_test_against_clean(category, n_runs=n_runs, model=model)
    lines.append("## Paired test vs. clean (RQ2, Jaccard)")
    lines.append("")
    if test is None:
        lines.append("clean description" if category == "clean" else "not enough paired descriptions to test")
    else:
        direction = "worse" if test["mean_delta"] < 0 else "better"
        significant = "significant" if test["p_value"] < 0.05 else "not significant"
        lines.append(
            f"- n pairs = {test['n_pairs']} (same desc_id, clean mean Jaccard vs. {category} mean Jaccard, each averaged over {n_runs} runs)")
        lines.append(
            f"- mean delta (jaccard_{category} - jaccard_clean) = {test['mean_delta']:+.3f}, median = {test['median_delta']:+.3f}")
        lines.append(
            f"- {test['n_worse']} descriptions worse than clean, {test['n_better']} better, {test['n_tied']} tied")
        lines.append(
            f"- Wilcoxon: statistic={test['statistic']:.1f}, p={test['p_value']:.4f} ({significant} at alpha=0.05)")
        lines.append(f"- on average this category performs {direction} than clean, and the difference is {significant}")
    lines.append("")

    trace_test = paired_test_against_clean(category, n_runs=n_runs, model=model, metric="trace_similarity_mean")
    lines.append("## Paired test vs. clean (trace similarity)")
    lines.append("")
    if trace_test is None:
        lines.append("clean description" if category == "clean" else "not enough paired descriptions to test")
    else:
        direction = "worse" if trace_test["mean_delta"] < 0 else "better"
        significant = "significant" if trace_test["p_value"] < 0.05 else "not significant"
        lines.append(
            f"- n pairs = {trace_test['n_pairs']} (same desc_id, clean mean trace similarity vs. {category} mean trace similarity, each averaged over {n_runs} runs)")
        lines.append(
            f"- mean delta (trace_{category} - trace_clean) = {trace_test['mean_delta']:+.3f}, median = {trace_test['median_delta']:+.3f}")
        lines.append(
            f"- {trace_test['n_worse']} descriptions worse than clean, {trace_test['n_better']} better, {trace_test['n_tied']} tied")
        lines.append(
            f"- Wilcoxon: statistic={trace_test['statistic']:.1f}, p={trace_test['p_value']:.4f} ({significant} at alpha=0.05)")
        lines.append(
            f"- on average this category performs {direction} than clean on trace similarity, and the difference is {significant}")
    lines.append("")

    output_path = output_dir / f"{category}.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_all_pet_injected_reports(
        n_runs: int = config.PET_INJECTED_N_RUNS,
        model: str = config.AUTOBPMN_DEFAULT_MODEL,
) -> List[Path]:
    category_dirs = sorted(p for p in config.PET_INJECTED_DIR.iterdir() if p.is_dir())
    written = []
    for category_dir in category_dirs:
        path = write_pet_injected_report(category_dir.name, n_runs=n_runs, model=model)
        print(f"wrote {path}")
        written.append(path)
    return written


def evaluate_deterministic_pipeline(
        category: str,
        config_name: str,
        n_runs: int = config.PET_INJECTED_N_RUNS,
        model: str = config.AUTOBPMN_DEFAULT_MODEL,
) -> Dict[str, dict]:
    from .dataset_generation import DETERMINISTIC_PIPELINE_DESCRIPTIONS_DIR

    description_dir = DETERMINISTIC_PIPELINE_DESCRIPTIONS_DIR / category / config_name
    generated_dir = config.dataset_generated_dir("deterministic_pipeline", f"pet_injected/{category}/{config_name}",
                                                 model=model)
    description_ids = sorted(p.stem for p in description_dir.glob("*.txt"))

    results: Dict[str, dict] = {}
    for desc_id in description_ids:
        gt_path = ground_truth_path_for_description("pet", desc_id)
        if gt_path is None:
            print(f"no ground-truth file found for description '{desc_id}'")
            continue
        gt_labels = extract_cpee_labels(gt_path)

        jaccards = []
        precisions = []
        recalls = []
        trace_similarities = []
        run_details = []
        failed_runs = 0
        for run_index in range(1, n_runs + 1):
            gen_path = generated_dir / f"run_{run_index}" / f"{desc_id}.xml"
            generation_failed = not gen_path.exists()
            if generation_failed:
                failed_runs += 1
            gen_labels = extract_cpee_labels(gen_path) if gen_path.exists() else []

            metrics = calculate_metrics(gt_labels, gen_labels)
            jaccards.append(metrics["jaccard"])
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])

            trace_similarity = None if generation_failed else get_similarity(gt_path, gen_path, sim_type="trace")
            if trace_similarity is not None:
                trace_similarities.append(trace_similarity)

            run_details.append({
                "run_index": run_index,
                "generation_failed": generation_failed,
                "matches": metrics["matches"],
                "unmatched_gt": metrics["unmatched_gt"],
                "unmatched_gen": metrics["unmatched_gen"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "jaccard": metrics["jaccard"],
                "trace_similarity": trace_similarity,
            })

        trace_similarity_mean = statistics.mean(trace_similarities) if trace_similarities else 0.0
        trace_similarity_sd = statistics.pstdev(trace_similarities) if len(trace_similarities) > 1 else 0.0

        results[desc_id] = {
            "n_runs": n_runs,
            "failed_runs": failed_runs,
            "jaccard_mean": statistics.mean(jaccards),
            "jaccard_sd": statistics.pstdev(jaccards) if len(jaccards) > 1 else 0.0,
            "precision_mean": statistics.mean(precisions),
            "recall_mean": statistics.mean(recalls),
            "trace_similarity_mean": trace_similarity_mean,
            "trace_similarity_sd": trace_similarity_sd,
            "runs": run_details,
        }
    return results


def paired_test_vs_pipeline_baseline(
        category: str,
        config_name: str,
        n_runs: int = config.PET_INJECTED_N_RUNS,
        model: str = config.AUTOBPMN_DEFAULT_MODEL,
        metric: str = "jaccard_mean",
) -> Optional[dict]:
    if config_name == "baseline":
        return None

    from scipy.stats import wilcoxon

    baseline_results = evaluate_deterministic_pipeline(category, "baseline", n_runs=n_runs, model=model)
    config_results = evaluate_deterministic_pipeline(category, config_name, n_runs=n_runs, model=model)

    shared_ids = sorted(set(baseline_results) & set(config_results))
    deltas = [config_results[d][metric] - baseline_results[d][metric] for d in shared_ids]

    if len(deltas) < 2 or all(d == 0 for d in deltas):
        return None

    statistic, p_value = wilcoxon(deltas)
    return {
        "n_pairs": len(deltas),
        "mean_delta": statistics.mean(deltas),
        "median_delta": statistics.median(deltas),
        "n_worse": sum(1 for d in deltas if d < 0),
        "n_better": sum(1 for d in deltas if d > 0),
        "n_tied": sum(1 for d in deltas if d == 0),
        "statistic": float(statistic),
        "p_value": float(p_value),
    }


def write_deterministic_pipeline_report(
        category: str,
        output_dir: Optional[Path] = None,
        n_runs: int = config.PET_INJECTED_N_RUNS,
        detail_config: str = "full",
        detail_run_index: int = 1,
        model: str = config.AUTOBPMN_DEFAULT_MODEL,
) -> Path:
    from .dataset_generation import DETERMINISTIC_PIPELINE_DESCRIPTIONS_DIR

    output_dir = (config.results_dir(model) / "deterministic_pipeline") if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    category_dir = DETERMINISTIC_PIPELINE_DESCRIPTIONS_DIR / category
    config_names = sorted(p.name for p in category_dir.iterdir() if p.is_dir())

    lines: List[str] = []
    lines.append(f"# deterministic_pipeline/pet_injected/{category} -- RQ3 evaluation against PET ground truth")
    lines.append("")
    lines.append(f"model = `{model}`, configs = {config_names}")
    lines.append("")
    lines.append(
        "| config | n | jaccard | precision | recall | trace similarity | failed runs | jaccard delta vs baseline | jaccard wilcoxon p | jaccard significant | trace delta vs baseline | trace wilcoxon p | trace significant |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    all_results: Dict[str, Dict[str, dict]] = {}
    for config_name in config_names:
        results = evaluate_deterministic_pipeline(category, config_name, n_runs=n_runs, model=model)
        all_results[config_name] = results
        n = len(results)
        total_runs = n * n_runs
        total_failed_runs = sum(r["failed_runs"] for r in results.values())

        jaccard_means = [r["jaccard_mean"] for r in results.values()]
        precision_means = [r["precision_mean"] for r in results.values()]
        recall_means = [r["recall_mean"] for r in results.values()]
        trace_similarity_means = [r["trace_similarity_mean"] for r in results.values()]
        mean_jaccard = statistics.mean(jaccard_means) if jaccard_means else 0.0
        mean_precision = statistics.mean(precision_means) if precision_means else 0.0
        mean_recall = statistics.mean(recall_means) if recall_means else 0.0
        mean_trace_similarity = statistics.mean(trace_similarity_means) if trace_similarity_means else 0.0

        if config_name == "baseline":
            lines.append(
                f"| {config_name} | {n} | {mean_jaccard:.3f} | {mean_precision:.3f} | {mean_recall:.3f} | {mean_trace_similarity:.3f} | {total_failed_runs}/{total_runs} | (reference) | -- | -- | (reference) | -- | -- |")
        else:
            test = paired_test_vs_pipeline_baseline(category, config_name, n_runs=n_runs, model=model)
            trace_test = paired_test_vs_pipeline_baseline(category, config_name, n_runs=n_runs, model=model,
                                                          metric="trace_similarity_mean")

            if test is None:
                jaccard_cols = "-- | -- | not enough pairs"
            else:
                significant = "yes" if test["p_value"] < 0.05 else "no"
                jaccard_cols = f"{test['mean_delta']:+.3f} | {test['p_value']:.4f} | {significant}"

            if trace_test is None:
                trace_cols = "-- | -- | not enough pairs"
            else:
                trace_significant = "yes" if trace_test["p_value"] < 0.05 else "no"
                trace_cols = f"{trace_test['mean_delta']:+.3f} | {trace_test['p_value']:.4f} | {trace_significant}"

            lines.append(
                f"| {config_name} | {n} | {mean_jaccard:.3f} | {mean_precision:.3f} | {mean_recall:.3f} | {mean_trace_similarity:.3f} | "
                f"{total_failed_runs}/{total_runs} | {jaccard_cols} | {trace_cols} |"
            )
    lines.append("")

    lines.append(
        f"## Label detail per description ('{detail_config}' config, run {detail_run_index} of {n_runs}, as an example)")
    lines.append("")
    detail_results = all_results.get(detail_config, {})
    for desc_id, r in detail_results.items():
        lines.append(f"### {desc_id}")
        run = find_run_detail(r["runs"], detail_run_index)
        if run is None or run["generation_failed"]:
            lines.append("")
            lines.append(f"(no generated XML for run {detail_run_index} -- AutoBPMN generation failed)")
            lines.append("")
            continue
        if run["trace_similarity"] is not None:
            lines.append("")
            lines.append(f"trace similarity = {run['trace_similarity']:.3f}")
        if run["matches"]:
            lines.append("")
            lines.append("MATCHED:")
            for gt_lbl, gen_lbl, score in run["matches"]:
                lines.append(f"- [{score:.2f}] GT `{gt_lbl}` <-> GEN `{gen_lbl}`")
        if run["unmatched_gt"]:
            lines.append("")
            lines.append("MISSED (in ground truth, not generated -- false negatives):")
            for lbl in run["unmatched_gt"]:
                lines.append(f"- {lbl}")
        if run["unmatched_gen"]:
            lines.append("")
            lines.append("EXTRA (generated, not in ground truth -- false positives):")
            for lbl in run["unmatched_gen"]:
                lines.append(f"- {lbl}")
        lines.append("")

    output_path = output_dir / f"{category}.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_all_deterministic_pipeline_reports(
        n_runs: int = config.PET_INJECTED_N_RUNS,
        model: str = config.AUTOBPMN_DEFAULT_MODEL,
) -> List[Path]:
    from .dataset_generation import DETERMINISTIC_PIPELINE_DESCRIPTIONS_DIR

    category_dirs = sorted(p for p in DETERMINISTIC_PIPELINE_DESCRIPTIONS_DIR.iterdir() if p.is_dir())
    written = []
    for category_dir in category_dirs:
        path = write_deterministic_pipeline_report(category_dir.name, n_runs=n_runs, model=model)
        print(f"wrote {path}")
        written.append(path)
    return written
