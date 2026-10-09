import argparse

from input_engineering import config
from input_engineering.pipeline import complete
from input_engineering.dataset_generation import (
    load_descriptions,
    generate_dataset_config,
    build_injected_pet_set,
    save_injected_pet_set,
    generate_pet_injected,
    build_pipeline_injected,
    save_pipeline_injected,
    generate_pipeline_injected,
    generate_mermaid_baseline_vs_full,
    generate_mermaid_pet_injected,
    generate_mermaid_pipeline_injected,
    convert_mermaid_dataset,
)
from input_engineering.run_evaluation import print_summary
from input_engineering.injected_evaluation import write_all_pet_injected_reports, \
    write_all_deterministic_pipeline_reports

MODEL_CHOICES = {
    "gemini": config.AUTOBPMN_DEFAULT_MODEL,
    "llama": config.LOCAL_LLM_MODEL,
}

AUTOBPMN_CHOICES = ["gemini"]


def cmd_generate(args: argparse.Namespace) -> None:
    descriptions = load_descriptions(args.dataset)
    suite = complete()
    names = args.configs or list(suite.keys())
    model = MODEL_CHOICES[args.model]
    print(f"Generating {len(descriptions)} description(s) x {len(names)} config(s) x {args.n_runs} run(s) "
          f"= {len(descriptions) * len(names) * args.n_runs} AutoBPMN calls, model={model}.")
    for name in names:
        generate_dataset_config(args.dataset, name, suite[name], descriptions, n_runs=args.n_runs, model=model)


def cmd_evaluate(args: argparse.Namespace) -> None:
    suite = complete()
    names = args.configs or list(suite.keys())
    print_summary(args.dataset, names, n_runs=args.n_runs, model=MODEL_CHOICES[args.model])


def cmd_inject_pet(args: argparse.Namespace) -> None:
    injected = build_injected_pet_set(seed=args.seed, target_edit_rate=args.target_edit_rate)
    save_injected_pet_set(injected)


def cmd_generate_injected(args: argparse.Namespace) -> None:
    generate_pet_injected(n_runs=args.n_runs, categories=args.categories, model=MODEL_CHOICES[args.model])


def cmd_evaluate_injected(args: argparse.Namespace) -> None:
    write_all_pet_injected_reports(n_runs=args.n_runs, model=MODEL_CHOICES[args.model])


def cmd_build_pipeline_injected(args: argparse.Namespace) -> None:
    preprocessed = build_pipeline_injected()
    save_pipeline_injected(preprocessed)


def cmd_generate_pipeline_injected(args: argparse.Namespace) -> None:
    generate_pipeline_injected(n_runs=args.n_runs, categories=args.categories, model=MODEL_CHOICES[args.model])


def cmd_evaluate_pipeline_injected(args: argparse.Namespace) -> None:
    write_all_deterministic_pipeline_reports(n_runs=args.n_runs, model=MODEL_CHOICES[args.model])


def cmd_generate_mermaid_local(args: argparse.Namespace) -> None:
    generate_mermaid_baseline_vs_full(n_runs=args.n_runs)


def cmd_generate_mermaid_injected(args: argparse.Namespace) -> None:
    generate_mermaid_pet_injected(categories=args.categories, n_runs=args.n_runs)


def cmd_generate_mermaid_pipeline_injected(args: argparse.Namespace) -> None:
    generate_mermaid_pipeline_injected(categories=args.categories, n_runs=args.n_runs)


def cmd_convert_mermaid(args: argparse.Namespace) -> None:
    convert_mermaid_dataset(args.dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_generate = sub.add_parser("generate", help="Generate CPEE XML with AutoBPMN")
    p_generate.add_argument("--dataset", choices=config.DATASETS, default="pet")
    p_generate.add_argument("--configs", nargs="*", default=None)
    p_generate.add_argument("--n-runs", type=int, default=config.N_RUNS)
    p_generate.add_argument("--model", choices=AUTOBPMN_CHOICES, default="gemini")
    p_generate.set_defaults(func=cmd_generate)

    p_evaluate = sub.add_parser("evaluate", help="Score generated XML")
    p_evaluate.add_argument("--dataset", choices=config.DATASETS, default="pet")
    p_evaluate.add_argument("--configs", nargs="*", default=None)
    p_evaluate.add_argument("--n-runs", type=int, default=config.N_RUNS)
    p_evaluate.add_argument("--model", choices=list(MODEL_CHOICES), default="gemini")
    p_evaluate.set_defaults(func=cmd_evaluate)

    p_inject = sub.add_parser("inject-pet", help="Build noised PET descriptions")
    p_inject.add_argument("--target-edit-rate", type=float, default=config.NOISE_TARGET_EDIT_RATE)
    p_inject.add_argument("--seed", type=int, default=42)
    p_inject.set_defaults(func=cmd_inject_pet)

    p_generate_injected = sub.add_parser("generate-injected", help="RQ2: generate from noised PET")
    p_generate_injected.add_argument("--n-runs", type=int, default=config.PET_INJECTED_N_RUNS)
    p_generate_injected.add_argument("--categories", nargs="*", default=None)
    p_generate_injected.add_argument("--model", choices=AUTOBPMN_CHOICES, default="gemini")
    p_generate_injected.set_defaults(func=cmd_generate_injected)

    p_evaluate_injected = sub.add_parser("evaluate-injected", help="RQ2: score noised PET")
    p_evaluate_injected.add_argument("--n-runs", type=int, default=config.PET_INJECTED_N_RUNS)
    p_evaluate_injected.add_argument("--model", choices=list(MODEL_CHOICES), default="gemini")
    p_evaluate_injected.set_defaults(func=cmd_evaluate_injected)

    p_build_pipeline_injected = sub.add_parser("build-pipeline-injected", help="RQ3: run pipeline on noised PET")
    p_build_pipeline_injected.set_defaults(func=cmd_build_pipeline_injected)

    p_generate_pipeline_injected = sub.add_parser("generate-pipeline-injected", help="RQ3: generate from pipeline output")
    p_generate_pipeline_injected.add_argument("--n-runs", type=int, default=config.PET_INJECTED_N_RUNS)
    p_generate_pipeline_injected.add_argument("--categories", nargs="*", default=None)
    p_generate_pipeline_injected.add_argument("--model", choices=AUTOBPMN_CHOICES, default="gemini")
    p_generate_pipeline_injected.set_defaults(func=cmd_generate_pipeline_injected)

    p_evaluate_pipeline_injected = sub.add_parser("evaluate-pipeline-injected", help="RQ3: score pipeline output")
    p_evaluate_pipeline_injected.add_argument("--n-runs", type=int, default=config.PET_INJECTED_N_RUNS)
    p_evaluate_pipeline_injected.add_argument("--model", choices=list(MODEL_CHOICES), default="gemini")
    p_evaluate_pipeline_injected.set_defaults(func=cmd_evaluate_pipeline_injected)

    p_generate_mermaid_local = sub.add_parser("generate-mermaid-local", help="Local llama: baseline vs full")
    p_generate_mermaid_local.add_argument("--n-runs", type=int, default=1)
    p_generate_mermaid_local.set_defaults(func=cmd_generate_mermaid_local)

    p_generate_mermaid_injected = sub.add_parser("generate-mermaid-injected", help="Local llama: RQ2")
    p_generate_mermaid_injected.add_argument("--categories", nargs="*", default=None)
    p_generate_mermaid_injected.add_argument("--n-runs", type=int, default=1)
    p_generate_mermaid_injected.set_defaults(func=cmd_generate_mermaid_injected)

    p_generate_mermaid_pipeline_injected = sub.add_parser("generate-mermaid-pipeline-injected", help="Local llama: RQ3")
    p_generate_mermaid_pipeline_injected.add_argument("--categories", nargs="*", default=None)
    p_generate_mermaid_pipeline_injected.add_argument("--n-runs", type=int, default=1)
    p_generate_mermaid_pipeline_injected.set_defaults(func=cmd_generate_mermaid_pipeline_injected)

    p_convert_mermaid = sub.add_parser("convert-mermaid", help="Convert .mmd to CPEE XML")
    p_convert_mermaid.add_argument("--dataset", choices=["pet_injected", "deterministic_pipeline", "pet"], default="pet_injected")
    p_convert_mermaid.set_defaults(func=cmd_convert_mermaid)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
