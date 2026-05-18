import argparse
from typing import Dict, List, Optional

from tqdm import tqdm

from semantic_similarity_eval import config
from semantic_similarity_eval.utils.io import dataset_result_dir, ensure_dir, read_csv, records_to_csv, write_json
from semantic_similarity_eval.utils.logging_utils import setup_logging
from semantic_similarity_eval.utils.metrics import auc_scores, binary_metrics, cosine_matrix, make_thresholds, metric_summary
from semantic_similarity_eval.utils.modeling import clear_cuda_cache, encode_texts, get_device, load_embedding_model, set_seed
from semantic_similarity_eval.utils.text_normalize import split_sentences


def parse_args():
    parser = argparse.ArgumentParser(description="Run pipeline/refinement ablation study.")
    parser.add_argument("--dataset", choices=list(config.DATASETS), default=None)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dataset_names(dataset: Optional[str]) -> List[str]:
    return [dataset] if dataset else list(config.DATASETS.keys())


def as_float(row: Dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def evaluate_variant(dataset: str, dataset_type: str, rows: List[Dict], variant: str, score_key: str, label_key: str) -> Dict:
    thresholds = make_thresholds(
        config.ANALYSIS["threshold_min"],
        config.ANALYSIS["threshold_max"],
        config.ANALYSIS["threshold_step"],
    )
    y_true = [int(float(row[label_key])) for row in rows]
    scores = [float(row[score_key]) for row in rows]
    summary = metric_summary(y_true, scores, thresholds)
    return {
        "dataset": dataset,
        "dataset_type": dataset_type,
        "ablation_family": "refinement_pipeline",
        "variant": variant,
        "score_key": score_key,
        "label_key": label_key,
        "n": len(rows),
        "hhem_threshold": config.HHEM["threshold"],
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


def evaluate_variant_at_threshold(
    dataset: str,
    dataset_type: str,
    rows: List[Dict],
    variant: str,
    score_key: str,
    label_key: str,
    threshold: float,
    calibration_scope: str,
) -> Dict:
    import numpy as np

    y_true = np.asarray([int(float(row[label_key])) for row in rows], dtype=int)
    scores = np.asarray([float(row[score_key]) for row in rows], dtype=float)
    y_pred = (scores >= threshold).astype(int)
    metrics = binary_metrics(y_true, y_pred)
    metrics.update(auc_scores(y_true, scores))
    return {
        "dataset": dataset,
        "dataset_type": dataset_type,
        "ablation_family": "refinement_pipeline",
        "variant": variant,
        "score_key": score_key,
        "label_key": label_key,
        "n": len(rows),
        "calibration_scope": calibration_scope,
        "calibrated_threshold": float(threshold),
        "hhem_threshold": config.HHEM["threshold"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "best_f1": metrics["f1"],
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "tp": metrics["tp"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
    }


def refined_alignment_for_rows(rows: List[Dict], tokenizer, model, device: str) -> List[Dict]:
    pred_spans = []
    ref_spans = []
    pred_units = []
    ref_units = []
    for row in rows:
        pred = split_sentences(row.get("cleaned_prediction_statement", ""))
        ref = split_sentences(row.get("reference_answer_core", ""))
        pred = pred or [row.get("cleaned_prediction_statement", "")]
        ref = ref or [row.get("reference_answer_core", "")]
        pred_start = len(pred_units)
        ref_start = len(ref_units)
        pred_units.extend(pred)
        ref_units.extend(ref)
        pred_spans.append((pred_start, len(pred_units), len(pred)))
        ref_spans.append((ref_start, len(ref_units), len(ref)))

    pred_vectors = encode_texts(
        tokenizer,
        model,
        tqdm(pred_units, desc="refined_alignment_pred_units"),
        device,
        config.EMBEDDING["batch_size"],
        config.EMBEDDING["max_length"],
    )
    ref_vectors = encode_texts(
        tokenizer,
        model,
        tqdm(ref_units, desc="refined_alignment_ref_units"),
        device,
        config.EMBEDDING["batch_size"],
        config.EMBEDDING["max_length"],
    )

    enriched = []
    for row, pred_span, ref_span in zip(rows, pred_spans, ref_spans):
        pred_start, pred_end, pred_count = pred_span
        ref_start, ref_end, ref_count = ref_span
        matrix = cosine_matrix(pred_vectors[pred_start:pred_end], ref_vectors[ref_start:ref_end])
        matrix = matrix.clip(0.0, 1.0)
        precision = float(matrix.max(axis=1).mean()) if matrix.shape[0] else 0.0
        recall = float(matrix.max(axis=0).mean()) if matrix.shape[1] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        enriched.append(
            {
                **row,
                "refined_alignment_precision": precision,
                "refined_alignment_recall": recall,
                "refined_alignment_f1": float(f1),
                "refined_prediction_unit_count": pred_count,
                "refined_reference_unit_count": ref_count,
            }
        )
    return enriched


def add_short_answer_scores(rows: List[Dict], dataset_type: str) -> List[Dict]:
    if dataset_type != "short":
        return rows
    weights = config.IMPROVEMENT["short_hybrid_weights"]
    enriched = []
    for row in rows:
        exact = as_float(row, "normalized_exact_match")
        token = as_float(row, "token_f1")
        list_match = as_float(row, "list_set_match")
        cosine = max(0.0, min(1.0, as_float(row, "refined_global_cosine")))
        hybrid = (
            weights["exact_match"] * exact
            + weights["token_f1"] * token
            + weights.get("list_set_match", 0.0) * list_match
            + weights["cosine"] * cosine
        )
        no_exact_den = weights["token_f1"] + weights.get("list_set_match", 0.0) + weights["cosine"]
        no_token_den = weights["exact_match"] + weights.get("list_set_match", 0.0) + weights["cosine"]
        enriched.append(
            {
                **row,
                "short_answer_hybrid": hybrid,
                "short_hybrid_without_exact": (
                    weights["token_f1"] * token
                    + weights.get("list_set_match", 0.0) * list_match
                    + weights["cosine"] * cosine
                ) / no_exact_den if no_exact_den > 0 else 0.0,
                "short_hybrid_without_token_f1": (
                    weights["exact_match"] * exact
                    + weights.get("list_set_match", 0.0) * list_match
                    + weights["cosine"] * cosine
                ) / no_token_den if no_token_den > 0 else 0.0,
            }
        )
    return enriched


def variant_specs(dataset_type: str) -> List[tuple]:
    variants = [
        ("base", "global_cosine", "original_correct_label"),
        ("semantic_refinement", "refined_global_cosine", "refined_hhem_correct_label"),
        ("claim_chunk_alignment", "refined_alignment_f1", "final_correct_label"),
        ("label_override_only", "global_cosine", "final_correct_label"),
        ("all_refinements", "refined_global_cosine", "final_correct_label"),
    ]
    if dataset_type == "short":
        variants.extend(
            [
                ("short_answer_hybrid", "short_answer_hybrid", "final_correct_label"),
                ("short_hybrid_without_exact", "short_hybrid_without_exact", "final_correct_label"),
                ("short_hybrid_without_token_f1", "short_hybrid_without_token_f1", "final_correct_label"),
            ]
        )
    return variants


def run_for_dataset(dataset: str, tokenizer, model, device: str, overwrite: bool = False) -> Dict:
    result_dir = dataset_result_dir(config.RESULTS_DIR, dataset)
    output_dir = ensure_dir(config.RESULTS_REFINE_DIR / "refinement_ablation" / dataset)
    logger = setup_logging("refinement_ablation", output_dir / "refinement_ablation.log")
    rows_path = result_dir / "refined_similarity_scores.csv"
    detailed_path = output_dir / "refinement_ablation_scores.csv"
    summary_path = output_dir / "refinement_ablation_summary.csv"
    metadata_path = output_dir / "refinement_ablation_metadata.json"

    if detailed_path.exists() and summary_path.exists() and not overwrite:
        logger.info("Refinement ablation already exists for %s; use --overwrite to recompute.", dataset)
        rows = read_csv(detailed_path)
        dataset_type = config.DATASETS[dataset]["type"]
        return {"summary": read_csv(summary_path), "rows": rows, "variants": variant_specs(dataset_type)}

    rows = read_csv(rows_path)
    if not rows:
        raise FileNotFoundError(f"No refined scores found at {rows_path}. Run refine_evaluation.py first.")
    rows = refined_alignment_for_rows(rows, tokenizer, model, device)
    dataset_type = config.DATASETS[dataset]["type"]
    rows = add_short_answer_scores(rows, dataset_type)

    variants = variant_specs(dataset_type)
    summary = [
        evaluate_variant(dataset, dataset_type, rows, variant, score_key, label_key)
        for variant, score_key, label_key in variants
    ]
    records_to_csv(detailed_path, rows)
    records_to_csv(summary_path, summary)
    write_json(output_dir / "refinement_ablation_summary.json", summary)
    write_json(
        metadata_path,
        {
            "dataset": dataset,
            "dataset_type": dataset_type,
            "variants": [
                {
                    "variant": variant,
                    "score_key": score_key,
                    "label_key": label_key,
                }
                for variant, score_key, label_key in variants
            ],
            "note": (
                "Selected ablation variants focus on the main refinement pipeline: base, "
                "semantic refinement, label overrides, and all refinements together. "
                "Claim/chunk alignment evaluates answer-aware factual-unit matching on refined text."
            ),
        },
    )
    logger.info("Saved refinement ablation for %s to %s", dataset, output_dir)
    return {"summary": summary, "rows": rows, "variants": variants}


def save_type_calibrated_outputs(results: List[Dict]) -> None:
    if not results:
        return
    table_dir = ensure_dir(config.RESULTS_REFINE_DIR / "tables")
    dataset_rows = []
    pooled_rows = []
    for result in results:
        dataset_rows.extend(result["rows"])
    for dataset_type in ["short", "long"]:
        type_rows = [row for row in dataset_rows if row["dataset_type"] == dataset_type]
        if not type_rows:
            continue
        variants = variant_specs(dataset_type)
        for variant, score_key, label_key in variants:
            pooled_summary = metric_summary(
                [int(float(row[label_key])) for row in type_rows],
                [float(row[score_key]) for row in type_rows],
                make_thresholds(
                    config.ANALYSIS["threshold_min"],
                    config.ANALYSIS["threshold_max"],
                    config.ANALYSIS["threshold_step"],
                ),
            )
            threshold = pooled_summary["threshold"]
            pooled_rows.append(
                evaluate_variant_at_threshold(
                    dataset_type,
                    dataset_type,
                    type_rows,
                    variant,
                    score_key,
                    label_key,
                    threshold,
                    "dataset_type_pooled",
                )
            )
            for dataset in dataset_names(None):
                rows = [row for row in type_rows if row["dataset"] == dataset]
                if not rows:
                    continue
                pooled_rows.append(
                    evaluate_variant_at_threshold(
                        dataset,
                        dataset_type,
                        rows,
                        variant,
                        score_key,
                        label_key,
                        threshold,
                        "dataset_type",
                    )
                )
    records_to_csv(table_dir / "refinement_ablation_type_calibrated_summary.csv", pooled_rows)
    write_json(table_dir / "refinement_ablation_type_calibrated_summary.json", pooled_rows)


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = get_device(args.device)
    logger = setup_logging("refinement_ablation", config.RESULTS_REFINE_DIR / "refinement_ablation.log")
    logger.info("Loading embedding model from %s on %s", config.MODEL_PATHS["embedding_model"], device)
    tokenizer, model = load_embedding_model(config.MODEL_PATHS["embedding_model"], device, config.TORCH_DTYPE)
    all_rows = []
    results = []
    for dataset in dataset_names(args.dataset):
        result = run_for_dataset(dataset, tokenizer, model, device, args.overwrite)
        all_rows.extend(result["summary"])
        results.append(result)
    if all_rows:
        table_dir = ensure_dir(config.RESULTS_REFINE_DIR / "tables")
        records_to_csv(table_dir / "refinement_ablation_summary.csv", all_rows)
        write_json(table_dir / "refinement_ablation_summary.json", all_rows)
        save_type_calibrated_outputs(results)
    clear_cuda_cache()


if __name__ == "__main__":
    main()
