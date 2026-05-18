import argparse
from typing import Dict, List, Optional

from semantic_similarity_eval import config
from semantic_similarity_eval.utils.io import dataset_result_dir, ensure_dir, read_csv, records_to_csv, write_json
from semantic_similarity_eval.utils.logging_utils import setup_logging
from semantic_similarity_eval.utils.metrics import make_thresholds, metric_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run component ablation study over existing metrics.")
    parser.add_argument("--dataset", choices=list(config.DATASETS), default=None)
    parser.add_argument("--device", default=config.DEVICE, help="Kept for pipeline compatibility; not used.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dataset_names(dataset: Optional[str]) -> List[str]:
    return [dataset] if dataset else list(config.DATASETS.keys())


def as_float(row: Dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def as_int(row: Dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mean_score(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def ablation_score(row: Dict, method: str) -> float:
    exact = as_float(row, "normalized_exact_match")
    token = as_float(row, "token_f1")
    cosine = clip01(as_float(row, "global_cosine"))
    alignment = clip01(as_float(row, "alignment_f1"))
    short_hybrid = clip01(as_float(row, "short_hybrid_score"))
    type_strategy = clip01(as_float(row, "type_strategy_score"))

    if method == "global_cosine_only":
        return cosine
    if method == "token_f1_only":
        return token
    if method == "exact_match_only":
        return exact
    if method == "alignment_only":
        return alignment
    if method == "short_hybrid_full":
        return short_hybrid
    if method == "type_strategy_full":
        return type_strategy
    if method == "hybrid_without_exact_match":
        return mean_score([token, cosine])
    if method == "hybrid_without_token_f1":
        return mean_score([exact, cosine])
    if method == "hybrid_without_cosine":
        return mean_score([exact, token])
    if method == "semantic_components_only":
        return mean_score([cosine, alignment])
    if method == "lexical_components_only":
        return mean_score([exact, token])
    if method == "all_components_uniform":
        return mean_score([exact, token, cosine, alignment])
    raise KeyError(f"Unknown ablation method: {method}")


def ablation_methods_for_dataset(dataset_type: str) -> List[str]:
    common = [
        "global_cosine_only",
        "token_f1_only",
        "exact_match_only",
        "alignment_only",
        "semantic_components_only",
        "lexical_components_only",
        "all_components_uniform",
    ]
    if dataset_type == "short":
        return [
            *common,
            "short_hybrid_full",
            "hybrid_without_exact_match",
            "hybrid_without_token_f1",
            "hybrid_without_cosine",
            "type_strategy_full",
        ]
    return [
        *common,
        "type_strategy_full",
    ]


def evaluate_method(dataset: str, dataset_type: str, rows: List[Dict], method: str) -> Dict:
    thresholds = make_thresholds(
        config.ANALYSIS["threshold_min"],
        config.ANALYSIS["threshold_max"],
        config.ANALYSIS["threshold_step"],
    )
    y_true = [as_int(row, "correct_label") for row in rows]
    scores = [ablation_score(row, method) for row in rows]
    summary = metric_summary(y_true, scores, thresholds)
    return {
        "dataset": dataset,
        "dataset_type": dataset_type,
        "ablation_family": "component_score",
        "method": method,
        "n": len(rows),
        "label_source": "hhem_correct_label",
        "hhem_threshold": config.HHEM["threshold"],
        "threshold_metric": "component_score",
        "threshold_min": config.ANALYSIS["threshold_min"],
        "threshold_max": config.ANALYSIS["threshold_max"],
        "threshold_step": config.ANALYSIS["threshold_step"],
        "best_threshold": summary["threshold"],
        "accuracy": summary["accuracy"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "best_f1": summary["f1"],
        "auroc": summary["auroc"],
        "auprc": summary["auprc"],
    }


def best_method(comparison: List[Dict]) -> Dict:
    return max(
        comparison,
        key=lambda row: (
            as_float(row, "best_f1"),
            as_float(row, "auprc"),
            as_float(row, "auroc"),
            -as_float(row, "best_threshold"),
        ),
    )


def run_ablation_for_dataset(dataset: str, device: str, overwrite: bool = False) -> List[Dict]:
    result_dir = dataset_result_dir(config.RESULTS_DIR, dataset)
    ablation_root = ensure_dir(config.RESULTS_REFINE_DIR / "ablation")
    dataset_ablation_dir = ensure_dir(ablation_root / dataset)
    logger = setup_logging("component_ablation", dataset_ablation_dir / "ablation.log")
    input_path = result_dir / "improved_similarity_scores.csv"
    results_path = dataset_ablation_dir / "ablation_results.csv"
    comparison_path = dataset_ablation_dir / "ablation_metric_comparison.csv"
    comparison_json_path = dataset_ablation_dir / "ablation_metric_comparison.json"
    metadata_path = dataset_ablation_dir / "ablation_metadata.json"

    if results_path.exists() and comparison_path.exists() and not overwrite:
        logger.info("Component ablation results already exist for %s; use --overwrite to recompute.", dataset)
        return read_csv(comparison_path)

    rows = read_csv(input_path)
    if not rows:
        raise FileNotFoundError(f"No improved scores found at {input_path}. Run improve_metric.py first.")
    dataset_type = config.DATASETS[dataset]["type"]
    methods = ablation_methods_for_dataset(dataset_type)
    comparison = [evaluate_method(dataset, dataset_type, rows, method) for method in methods]
    best = best_method(comparison)
    best_name = best["method"]
    best_threshold = as_float(best, "best_threshold")
    thresholds_by_method = {row["method"]: as_float(row, "best_threshold") for row in comparison}
    baseline_threshold = thresholds_by_method["global_cosine_only"]
    ablation_results = []

    for row in rows:
        sample_id = str(row["sample_id"])
        correct_label = as_int(row, "correct_label")
        baseline_score = ablation_score(row, "global_cosine_only")
        baseline_pred = int(baseline_score >= baseline_threshold)
        best_score = ablation_score(row, best_name)
        best_pred = int(best_score >= best_threshold)
        for method in methods:
            score = ablation_score(row, method)
            method_threshold = thresholds_by_method[method]
            predicted_label = int(score >= method_threshold)
            ablation_results.append(
                {
                    "dataset": dataset,
                    "dataset_type": dataset_type,
                    "sample_id": sample_id,
                    "ablation_family": "component_score",
                    "method": method,
                    "score": score,
                    "best_threshold": method_threshold,
                    "predicted_label": predicted_label,
                    "correct_label": correct_label,
                    "is_error": int(predicted_label != correct_label),
                    "hhem_threshold": as_float(row, "hhem_threshold", config.HHEM["threshold"]),
                    "hhem_correctness_score": as_float(row, "hhem_correctness_score"),
                    "baseline_global_cosine_score": baseline_score,
                    "baseline_global_cosine_threshold": baseline_threshold,
                    "baseline_global_cosine_predicted_label": baseline_pred,
                    "best_ablation_method": best_name,
                    "best_ablation_score": best_score,
                    "best_ablation_threshold": best_threshold,
                    "best_ablation_predicted_label": best_pred,
                }
            )

    records_to_csv(results_path, ablation_results)
    records_to_csv(comparison_path, comparison)
    write_json(comparison_json_path, comparison)
    write_json(
        metadata_path,
        {
            "dataset": dataset,
            "dataset_type": dataset_type,
            "ablation_family": "component_score",
            "input": str(input_path),
            "methods": methods,
            "best_method": best_name,
            "best_threshold": best_threshold,
            "hhem_threshold": config.HHEM["threshold"],
            "threshold_min": config.ANALYSIS["threshold_min"],
            "threshold_max": config.ANALYSIS["threshold_max"],
            "threshold_step": config.ANALYSIS["threshold_step"],
            "note": "No alternate embedding models are loaded; this ablates existing scoring components.",
        },
    )
    logger.info(
        "Component ablation saved for %s; best method=%s, threshold=%.2f, best_f1=%.3f",
        dataset,
        best_name,
        best_threshold,
        as_float(best, "best_f1"),
    )
    return comparison


def main():
    args = parse_args()
    all_comparisons = []
    for dataset in dataset_names(args.dataset):
        all_comparisons.extend(run_ablation_for_dataset(dataset, args.device, args.overwrite))
    if all_comparisons:
        table_dir = ensure_dir(config.RESULTS_REFINE_DIR / "tables")
        if args.dataset is None:
            records_to_csv(table_dir / "component_ablation_summary.csv", all_comparisons)
            write_json(table_dir / "component_ablation_summary.json", all_comparisons)
            records_to_csv(table_dir / "embedding_ablation_summary.csv", all_comparisons)
            write_json(table_dir / "embedding_ablation_summary.json", all_comparisons)
        else:
            records_to_csv(table_dir / f"component_ablation_summary_{args.dataset}.csv", all_comparisons)
            write_json(table_dir / f"component_ablation_summary_{args.dataset}.json", all_comparisons)
            records_to_csv(table_dir / f"embedding_ablation_summary_{args.dataset}.csv", all_comparisons)
            write_json(table_dir / f"embedding_ablation_summary_{args.dataset}.json", all_comparisons)


if __name__ == "__main__":
    main()
