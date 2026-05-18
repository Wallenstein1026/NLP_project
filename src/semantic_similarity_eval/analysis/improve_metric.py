import argparse
from typing import Dict, List, Optional

from tqdm import tqdm

from semantic_similarity_eval import config
from semantic_similarity_eval.analysis.analyze_similarity import load_joined_records
from semantic_similarity_eval.utils.io import dataset_result_dir, read_csv, records_to_csv, write_json
from semantic_similarity_eval.utils.logging_utils import setup_logging
from semantic_similarity_eval.utils.modeling import clear_cuda_cache, encode_texts, get_device, load_embedding_model, set_seed
from semantic_similarity_eval.utils.text_normalize import list_set_match, normalized_exact_match, split_sentences, token_f1


def parse_args():
    parser = argparse.ArgumentParser(description="Compute improved semantic similarity metrics.")
    parser.add_argument("--dataset", choices=list(config.DATASETS), default=None)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dataset_names(dataset: Optional[str]) -> List[str]:
    return [dataset] if dataset else list(config.DATASETS.keys())


def short_hybrid_score(row: Dict) -> float:
    from semantic_similarity_eval.utils.metrics import normalize_cosine_for_mixture

    weights = config.IMPROVEMENT["short_hybrid_weights"]
    exact = normalized_exact_match(row["raw_prediction"], row["true_answer"])
    tf1 = token_f1(row["raw_prediction"], row["true_answer"])
    list_match = list_set_match(row["raw_prediction"], row["true_answer"])
    cosine = normalize_cosine_for_mixture(row["global_cosine"])
    return float(
        weights["exact_match"] * exact
        + weights["token_f1"] * tf1
        + weights.get("list_set_match", 0.0) * list_match
        + weights["cosine"] * cosine
    )


def alignment_score(pred_embeddings, ref_embeddings) -> Dict[str, float]:
    import numpy as np
    from semantic_similarity_eval.utils.metrics import cosine_matrix

    if pred_embeddings.size == 0 or ref_embeddings.size == 0:
        return {"alignment_precision": 0.0, "alignment_recall": 0.0, "alignment_f1": 0.0}
    matrix = cosine_matrix(pred_embeddings, ref_embeddings)
    matrix = np.clip(matrix, 0.0, 1.0)
    precision = float(matrix.max(axis=1).mean()) if matrix.shape[0] else 0.0
    recall = float(matrix.max(axis=0).mean()) if matrix.shape[1] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "alignment_precision": precision,
        "alignment_recall": recall,
        "alignment_f1": float(f1),
    }


def compute_alignment_for_chunk(rows: List[Dict], tokenizer, model, device: str) -> List[Dict]:
    pred_spans = []
    ref_spans = []
    all_pred_sentences = []
    all_ref_sentences = []
    for row in rows:
        pred_text = row["raw_prediction"] if config.IMPROVEMENT["alignment_text_source"] == "answers" else row["merged_prediction_statement"]
        ref_text = row["true_answer"] if config.IMPROVEMENT["alignment_text_source"] == "answers" else row["merged_true_statement"]
        pred_sentences = split_sentences(pred_text)
        ref_sentences = split_sentences(ref_text)
        pred_start = len(all_pred_sentences)
        ref_start = len(all_ref_sentences)
        all_pred_sentences.extend(pred_sentences)
        all_ref_sentences.extend(ref_sentences)
        pred_spans.append((pred_start, len(all_pred_sentences), len(pred_sentences)))
        ref_spans.append((ref_start, len(all_ref_sentences), len(ref_sentences)))

    pred_vectors = encode_texts(
        tokenizer,
        model,
        tqdm(all_pred_sentences, desc="encode_alignment_predictions"),
        device,
        config.IMPROVEMENT["alignment_batch_size"],
        config.IMPROVEMENT["alignment_max_length"],
    )
    ref_vectors = encode_texts(
        tokenizer,
        model,
        tqdm(all_ref_sentences, desc="encode_alignment_references"),
        device,
        config.IMPROVEMENT["alignment_batch_size"],
        config.IMPROVEMENT["alignment_max_length"],
    )

    result_rows = []
    for row, pred_span, ref_span in zip(tqdm(rows, desc=f"alignment:{rows[0]['dataset'] if rows else 'dataset'}"), pred_spans, ref_spans):
        pred_start, pred_end, pred_count = pred_span
        ref_start, ref_end, ref_count = ref_span
        pred_vecs = pred_vectors[pred_start:pred_end]
        ref_vecs = ref_vectors[ref_start:ref_end]
        scores = alignment_score(pred_vecs, ref_vecs)
        result_rows.append(
            {
                "sample_id": row["sample_id"],
                "prediction_sentence_count": pred_count,
                "reference_sentence_count": ref_count,
                **scores,
            }
        )
    return result_rows


def compute_alignment_for_rows(rows: List[Dict], tokenizer, model, device: str) -> List[Dict]:
    chunk_size = config.IMPROVEMENT["alignment_chunk_size"]
    results = []
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start + chunk_size]
        results.extend(compute_alignment_for_chunk(chunk, tokenizer, model, device))
    return results


def evaluate_metric(rows: List[Dict], score_key: str) -> Dict:
    from semantic_similarity_eval.utils.metrics import make_thresholds, metric_summary

    thresholds = make_thresholds(
        config.ANALYSIS["threshold_min"],
        config.ANALYSIS["threshold_max"],
        config.ANALYSIS["threshold_step"],
    )
    y_true = [row["correct_label"] for row in rows]
    scores = [row[score_key] for row in rows]
    summary = metric_summary(y_true, scores, thresholds)
    return {
        "metric": score_key,
        "best_threshold": summary["threshold"],
        "accuracy": summary["accuracy"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "best_f1": summary["f1"],
        "auroc": summary["auroc"],
        "auprc": summary["auprc"],
    }


def run_for_dataset(dataset: str, tokenizer, model, device: str, overwrite: bool = False) -> Dict:
    result_dir = dataset_result_dir(config.RESULTS_DIR, dataset)
    logger = setup_logging("improve_metric", result_dir / "improve_metric.log")
    output_path = result_dir / "improved_similarity_scores.csv"
    comparison_path = result_dir / "metric_comparison.csv"
    metadata_path = result_dir / "improvement_metadata.json"
    if output_path.exists() and comparison_path.exists() and not overwrite:
        logger.info("Improved metrics already exist for %s; use --overwrite to recompute.", dataset)
        return {"rows": read_csv(output_path), "comparison": read_csv(comparison_path)}
    rows = load_joined_records(dataset)
    dataset_type = config.DATASETS[dataset]["type"]
    logger.info("Computing improved metrics for %s (%d rows)", dataset, len(rows))

    alignment_rows = compute_alignment_for_rows(rows, tokenizer, model, device)
    alignment_by_id = {str(row["sample_id"]): row for row in alignment_rows}

    improved_rows = []
    for row in rows:
        exact = normalized_exact_match(row["raw_prediction"], row["true_answer"])
        tf1 = token_f1(row["raw_prediction"], row["true_answer"])
        align = alignment_by_id[str(row["sample_id"])]
        hybrid = short_hybrid_score(row)
        type_strategy = hybrid if dataset_type == "short" else align["alignment_f1"]
        improved_rows.append(
            {
                **row,
                "normalized_exact_match": exact,
                "token_f1": tf1,
                "short_hybrid_score": hybrid,
                "alignment_precision": align["alignment_precision"],
                "alignment_recall": align["alignment_recall"],
                "alignment_f1": align["alignment_f1"],
                "prediction_sentence_count": align["prediction_sentence_count"],
                "reference_sentence_count": align["reference_sentence_count"],
                "type_strategy_score": type_strategy,
            }
        )

    metric_keys = ["global_cosine", "short_hybrid_score", "alignment_f1", "type_strategy_score"]
    comparison = []
    for key in metric_keys:
        row = {
            "dataset": dataset,
            "dataset_type": dataset_type,
            "n": len(improved_rows),
        }
        row.update(evaluate_metric(improved_rows, key))
        comparison.append(row)

    records_to_csv(output_path, improved_rows)
    records_to_csv(comparison_path, comparison)
    write_json(result_dir / "metric_comparison.json", comparison)
    write_json(
        metadata_path,
        {
            "dataset": dataset,
            "embedding_model": str(config.MODEL_PATHS["embedding_model"]),
            "short_hybrid_weights": config.IMPROVEMENT["short_hybrid_weights"],
            "alignment_text_source": config.IMPROVEMENT["alignment_text_source"],
            "completed_samples": len(improved_rows),
        },
    )
    logger.info("Saved improved metric comparison to %s", comparison_path)
    return {"rows": improved_rows, "comparison": comparison}


def save_global_comparison(all_comparisons: List[Dict]) -> None:
    records_to_csv(config.RESULTS_DIR / "metric_comparison_all.csv", all_comparisons)
    write_json(config.RESULTS_DIR / "metric_comparison_all.json", all_comparisons)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="paper")
        plot_rows = []
        for row in all_comparisons:
            for metric in ["auroc", "auprc", "best_f1"]:
                try:
                    value = float(row.get(metric))
                except (TypeError, ValueError):
                    continue
                if value != value:
                    continue
                plot_rows.append(
                    {
                        "dataset": row["dataset"],
                        "method": row["metric"],
                        "metric": metric.upper(),
                        "value": value,
                    }
                )
        if plot_rows:
            fig_dir = config.FIGURES_DIR
            fig_dir.mkdir(parents=True, exist_ok=True)
            for metric_name in ["AUROC", "AUPRC", "BEST_F1"]:
                subset = [row for row in plot_rows if row["metric"] == metric_name]
                plt.figure(figsize=(7.4, 3.8))
                sns.barplot(data=pd.DataFrame(subset), x="dataset", y="value", hue="method")
                plt.ylim(0, 1)
                plt.xlabel("Dataset")
                plt.ylabel(metric_name.replace("_", "-"))
                plt.title(f"Metric Comparison: {metric_name.replace('_', '-')}")
                plt.xticks(rotation=20, ha="right")
                plt.tight_layout()
                safe_name = metric_name.lower()
                plt.savefig(fig_dir / f"metric_comparison_{safe_name}.png", dpi=config.ANALYSIS["figure_dpi"])
                plt.savefig(fig_dir / f"metric_comparison_{safe_name}.pdf")
                plt.close()
    except Exception:
        return


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = get_device(args.device)
    logger = setup_logging("improve_metric", config.RESULTS_DIR / "improve_metric.log")
    logger.info("Loading embedding model from %s on %s", config.MODEL_PATHS["embedding_model"], device)
    tokenizer, model = load_embedding_model(config.MODEL_PATHS["embedding_model"], device, config.TORCH_DTYPE)
    all_comparisons = []
    for dataset in dataset_names(args.dataset):
        result = run_for_dataset(dataset, tokenizer, model, device, overwrite=args.overwrite)
        all_comparisons.extend(result["comparison"])
    if args.dataset is None:
        save_global_comparison(all_comparisons)
    clear_cuda_cache()


if __name__ == "__main__":
    main()
