import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from semantic_similarity_eval import config
from semantic_similarity_eval.utils.io import copy_if_exists, dataset_result_dir, ensure_dir, read_json, records_to_csv, read_csv, write_json, write_jsonl
from semantic_similarity_eval.utils.logging_utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Export interpretable failure cases for report analysis.")
    parser.add_argument("--dataset", choices=list(config.DATASETS), default=None)
    parser.add_argument("--export-report-ready", action="store_true")
    parser.add_argument("--report-ready-only", action="store_true")
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


def load_best_ablation_rows(dataset: str) -> Dict[str, Dict]:
    metadata = read_json(config.RESULTS_REFINE_DIR / "ablation" / dataset / "ablation_metadata.json", default={})
    best_method = metadata.get("best_method")
    rows = read_csv(config.RESULTS_REFINE_DIR / "ablation" / dataset / "ablation_results.csv")
    if not best_method or not rows:
        return {}
    return {
        str(row["sample_id"]): row
        for row in rows
        if row.get("method") == best_method
    }


def attach_ablation_context(dataset: str, rows: List[Dict]) -> List[Dict]:
    ablation_by_id = load_best_ablation_rows(dataset)
    if not ablation_by_id:
        return rows
    merged = []
    for row in rows:
        out = dict(row)
        ablation = ablation_by_id.get(str(row["sample_id"]))
        if ablation:
            out.update(
                {
                    "best_ablation_method": ablation.get("best_ablation_method") or ablation.get("method"),
                    "best_ablation_score": ablation.get("best_ablation_score") or ablation.get("score"),
                    "best_ablation_threshold": ablation.get("best_ablation_threshold") or ablation.get("best_threshold"),
                    "best_ablation_predicted_label": ablation.get("best_ablation_predicted_label") or ablation.get("predicted_label"),
                    "baseline_global_cosine_threshold": ablation.get("baseline_global_cosine_threshold"),
                    "baseline_global_cosine_predicted_label": ablation.get("baseline_global_cosine_predicted_label"),
                }
            )
        merged.append(out)
    return merged


def compact_case(row: Dict, failure_type: str, rank_score: float, threshold_context: Dict) -> Dict:
    return {
        "failure_type": failure_type,
        "dataset": row["dataset"],
        "dataset_type": row["dataset_type"],
        "sample_id": row["sample_id"],
        "question": row["question"],
        "prediction": row["raw_prediction"],
        "reference": row["true_answer"],
        "prediction_statement": row["merged_prediction_statement"],
        "reference_statement": row["merged_true_statement"],
        "correct_label": as_int(row, "correct_label"),
        "hhem_threshold": as_float(row, "hhem_threshold", config.HHEM["threshold"]),
        "global_cosine": as_float(row, "global_cosine"),
        "global_cosine_high_cutoff": threshold_context["global_cosine_high_cutoff"],
        "global_cosine_low_cutoff": threshold_context["global_cosine_low_cutoff"],
        "global_cosine_high_quantile": threshold_context["global_cosine_high_quantile"],
        "global_cosine_low_quantile": threshold_context["global_cosine_low_quantile"],
        "hhem_correctness_score": as_float(row, "hhem_correctness_score"),
        "token_f1": as_float(row, "token_f1"),
        "short_hybrid_score": as_float(row, "short_hybrid_score"),
        "alignment_f1": as_float(row, "alignment_f1"),
        "type_strategy_score": as_float(row, "type_strategy_score"),
        "best_ablation_method": row.get("best_ablation_method", ""),
        "best_ablation_score": row.get("best_ablation_score", ""),
        "best_ablation_threshold": row.get("best_ablation_threshold", ""),
        "best_ablation_predicted_label": row.get("best_ablation_predicted_label", ""),
        "baseline_global_cosine_threshold": row.get("baseline_global_cosine_threshold", ""),
        "baseline_global_cosine_predicted_label": row.get("baseline_global_cosine_predicted_label", ""),
        "prediction_length": as_float(row, "prediction_length"),
        "reference_length": as_float(row, "reference_length"),
        "rank_score": rank_score,
    }


def top_cases(
    rows: List[Dict],
    failure_type: str,
    rank_key: str,
    threshold_context: Dict,
    reverse: bool = True,
) -> List[Dict]:
    sorted_rows = sorted(rows, key=lambda row: as_float(row, rank_key), reverse=reverse)
    k = config.FAILURE_ANALYSIS["top_k_per_type"]
    return [compact_case(row, failure_type, as_float(row, rank_key), threshold_context) for row in sorted_rows[:k]]


def extract_failure_cases(dataset: str) -> Tuple[Dict[str, List[Dict]], Dict]:
    import numpy as np

    result_dir = dataset_result_dir(config.RESULTS_DIR, dataset)
    rows = read_csv(result_dir / "improved_similarity_scores.csv")
    if not rows:
        raise FileNotFoundError(f"No improved scores found for {dataset}. Run improve_metric.py first.")
    rows = attach_ablation_context(dataset, rows)

    cosines = np.asarray([as_float(row, "global_cosine") for row in rows], dtype=float)
    high_cut = float(np.quantile(cosines, config.FAILURE_ANALYSIS["high_similarity_quantile"]))
    low_cut = float(np.quantile(cosines, config.FAILURE_ANALYSIS["low_similarity_quantile"]))
    dataset_type = config.DATASETS[dataset]["type"]
    threshold_context = {
        "dataset": dataset,
        "dataset_type": dataset_type,
        "n": len(rows),
        "hhem_threshold": config.HHEM["threshold"],
        "global_cosine_high_quantile": config.FAILURE_ANALYSIS["high_similarity_quantile"],
        "global_cosine_high_cutoff": high_cut,
        "global_cosine_low_quantile": config.FAILURE_ANALYSIS["low_similarity_quantile"],
        "global_cosine_low_cutoff": low_cut,
        "lexical_token_f1_max": config.FAILURE_ANALYSIS["lexical_token_f1_max"],
        "semantic_ambiguity_token_f1_min": config.FAILURE_ANALYSIS["semantic_ambiguity_token_f1_min"],
        "long_reference_word_min": config.FAILURE_ANALYSIS["long_reference_word_min"],
        "top_k_per_type": config.FAILURE_ANALYSIS["top_k_per_type"],
    }

    high_similarity_incorrect = [
        row for row in rows
        if as_int(row, "correct_label") == 0 and as_float(row, "global_cosine") >= high_cut
    ]
    low_similarity_correct = [
        row for row in rows
        if as_int(row, "correct_label") == 1 and as_float(row, "global_cosine") <= low_cut
    ]
    lexical_variation = [
        row for row in rows
        if as_int(row, "correct_label") == 1
        and as_float(row, "token_f1") <= config.FAILURE_ANALYSIS["lexical_token_f1_max"]
        and as_float(row, "global_cosine") >= high_cut
    ]
    semantic_ambiguity = [
        row for row in rows
        if as_int(row, "correct_label") == 0
        and as_float(row, "token_f1") >= config.FAILURE_ANALYSIS["semantic_ambiguity_token_f1_min"]
        and as_float(row, "global_cosine") >= high_cut
    ]
    long_reasoning_complexity = [
        row for row in rows
        if dataset_type == "long"
        and as_float(row, "reference_length") >= config.FAILURE_ANALYSIS["long_reference_word_min"]
        and as_int(row, "correct_label") == 0
    ]

    # Additional failure categories
    hallucination_like = [
        row for row in rows
        if as_int(row, "correct_label") == 0
        and as_float(row, "global_cosine") >= high_cut
        and as_float(row, "token_f1") <= 0.1  # Very low lexical overlap
    ]

    paraphrase_correct = [
        row for row in rows
        if as_int(row, "correct_label") == 1
        and as_float(row, "global_cosine") >= high_cut
        and as_float(row, "token_f1") <= 0.5  # Low lexical but high semantic similarity
    ]
    ablation_best_error = [
        row for row in rows
        if row.get("best_ablation_predicted_label") not in (None, "")
        and as_int(row, "best_ablation_predicted_label") != as_int(row, "correct_label")
    ]
    cosine_wrong_ablation_right = [
        row for row in rows
        if row.get("baseline_global_cosine_predicted_label") not in (None, "")
        and row.get("best_ablation_predicted_label") not in (None, "")
        and as_int(row, "baseline_global_cosine_predicted_label") != as_int(row, "correct_label")
        and as_int(row, "best_ablation_predicted_label") == as_int(row, "correct_label")
    ]
    ablation_wrong_cosine_right = [
        row for row in rows
        if row.get("baseline_global_cosine_predicted_label") not in (None, "")
        and row.get("best_ablation_predicted_label") not in (None, "")
        and as_int(row, "baseline_global_cosine_predicted_label") == as_int(row, "correct_label")
        and as_int(row, "best_ablation_predicted_label") != as_int(row, "correct_label")
    ]

    cases = {
        "high_similarity_but_incorrect": top_cases(high_similarity_incorrect, "high_similarity_but_incorrect", "global_cosine", threshold_context, True),
        "low_similarity_but_correct": top_cases(low_similarity_correct, "low_similarity_but_correct", "global_cosine", threshold_context, False),
        "lexical_variation": top_cases(lexical_variation, "lexical_variation", "token_f1", threshold_context, False),
        "semantic_ambiguity": top_cases(semantic_ambiguity, "semantic_ambiguity", "global_cosine", threshold_context, True),
        "long_form_reasoning_complexity": top_cases(long_reasoning_complexity, "long_form_reasoning_complexity", "reference_length", threshold_context, True),
        "hallucination_like": top_cases(hallucination_like, "hallucination_like", "global_cosine", threshold_context, True),
        "paraphrase_correct": top_cases(paraphrase_correct, "paraphrase_correct", "token_f1", threshold_context, False),
        "ablation_best_error": top_cases(ablation_best_error, "ablation_best_error", "best_ablation_score", threshold_context, True),
        "cosine_wrong_ablation_right": top_cases(cosine_wrong_ablation_right, "cosine_wrong_ablation_right", "best_ablation_score", threshold_context, True),
        "ablation_wrong_cosine_right": top_cases(ablation_wrong_cosine_right, "ablation_wrong_cosine_right", "global_cosine", threshold_context, True),
    }
    threshold_context["case_counts"] = {failure_type: len(cases) for failure_type, cases in cases.items()}
    return cases, threshold_context


def run_for_dataset(dataset: str) -> List[Dict]:
    result_dir = dataset_result_dir(config.RESULTS_DIR, dataset)
    logger = setup_logging("analyze_failures", result_dir / "failures.log")
    failure_dir = ensure_dir(result_dir / "failure_cases")
    cases_by_type, threshold_metadata = extract_failure_cases(dataset)
    all_cases = []
    for failure_type, cases in cases_by_type.items():
        records_to_csv(failure_dir / f"{failure_type}.csv", cases)
        write_jsonl(failure_dir / f"{failure_type}.jsonl", cases)
        all_cases.extend(cases)
    records_to_csv(failure_dir / "all_failure_cases.csv", all_cases)
    write_jsonl(failure_dir / "all_failure_cases.jsonl", all_cases)
    write_json(failure_dir / "failure_analysis_metadata.json", threshold_metadata)
    logger.info(
        "Saved %d failure cases for %s; HHEM threshold=%.2f, high cosine cutoff=%.4f, low cosine cutoff=%.4f",
        len(all_cases),
        dataset,
        threshold_metadata["hhem_threshold"],
        threshold_metadata["global_cosine_high_cutoff"],
        threshold_metadata["global_cosine_low_cutoff"],
    )
    return all_cases


def export_report_ready(all_cases: List[Dict]) -> None:
    report_dir = ensure_dir(config.REPORT_READY_DIR)
    records_to_csv(report_dir / "failure_cases_all.csv", all_cases)
    write_jsonl(report_dir / "failure_cases_all.jsonl", all_cases)

    copy_if_exists(config.RESULTS_DIR / "summary_metrics.csv", report_dir / "summary_metrics.csv")
    copy_if_exists(config.RESULTS_DIR / "short_vs_long_summary.csv", report_dir / "short_vs_long_summary.csv")
    copy_if_exists(config.RESULTS_DIR / "metric_comparison_all.csv", report_dir / "metric_comparison_all.csv")
    copy_if_exists(config.RESULTS_DIR / "refined_summary_metrics.csv", report_dir / "refined_summary_metrics.csv")
    copy_if_exists(config.RESULTS_DIR / "refined_short_vs_long_summary.csv", report_dir / "refined_short_vs_long_summary.csv")
    copy_if_exists(config.RESULTS_DIR / "refined_metric_comparison_all.csv", report_dir / "refined_metric_comparison_all.csv")
    copy_if_exists(config.RESULTS_DIR / "label_audit_short.csv", report_dir / "label_audit_short.csv")
    copy_if_exists(config.RESULTS_DIR / "reference_core_audit_long.csv", report_dir / "reference_core_audit_long.csv")
    copy_if_exists(config.RESULTS_DIR / "reference_core_manual_review_50.csv", report_dir / "reference_core_manual_review_50.csv")
    copy_if_exists(Path(config.__file__), report_dir / "config.py")
    metadata = []
    for dataset in config.DATASETS:
        path = config.RESULTS_DIR / dataset / "failure_cases" / "failure_analysis_metadata.json"
        if path.exists():
            import json

            with path.open("r", encoding="utf-8") as f:
                metadata.append(json.load(f))
    if metadata:
        write_json(report_dir / "failure_analysis_metadata.json", metadata)

    figures_dst = ensure_dir(report_dir / "figures")
    if config.FIGURES_DIR.exists():
        for path in config.FIGURES_DIR.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".pdf"}:
                rel = path.relative_to(config.FIGURES_DIR)
                dst = figures_dst / rel
                ensure_dir(dst.parent)
                shutil.copy2(path, dst)


def load_existing_failure_cases(dataset: Optional[str] = None) -> List[Dict]:
    cases = []
    names = dataset_names(dataset)
    for name in names:
        path = config.RESULTS_DIR / name / "failure_cases" / "all_failure_cases.csv"
        cases.extend(read_csv(path))
    return cases


def main():
    args = parse_args()
    logger = setup_logging("analyze_failures", config.RESULTS_DIR / "analyze_failures.log")
    all_cases = []
    if args.report_ready_only:
        all_cases = load_existing_failure_cases(args.dataset)
    else:
        for dataset in dataset_names(args.dataset):
            all_cases.extend(run_for_dataset(dataset))
    if args.dataset is None or args.export_report_ready or args.report_ready_only:
        export_report_ready(all_cases)
        logger.info("Exported report-ready artifacts to %s", config.REPORT_READY_DIR)


if __name__ == "__main__":
    main()
