import argparse
import math
import os
from typing import Dict, List, Optional

from semantic_similarity_eval import config
from semantic_similarity_eval.utils.io import dataset_result_dir, ensure_dir, load_pickle, records_to_csv, read_jsonl, write_json
from semantic_similarity_eval.utils.logging_utils import setup_logging
from semantic_similarity_eval.utils.metrics import cosine_similarity
from semantic_similarity_eval.utils.text_normalize import normalized_exact_match, token_f1, word_count


_BLEU_METRIC = None
_BLEU_UNAVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze cosine similarity against correctness labels.")
    parser.add_argument("--dataset", choices=list(config.DATASETS), default=None)
    return parser.parse_args()


def dataset_names(dataset: Optional[str]) -> List[str]:
    return [dataset] if dataset else list(config.DATASETS.keys())


def compute_bleu(prediction: str, reference: str) -> float:
    global _BLEU_METRIC, _BLEU_UNAVAILABLE
    if _BLEU_UNAVAILABLE:
        return 0.0
    try:
        from evaluate import load

        if _BLEU_METRIC is None:
            _BLEU_METRIC = load("bleu")
        return float(_BLEU_METRIC.compute(predictions=[prediction], references=[reference])["bleu"])
    except Exception:
        _BLEU_UNAVAILABLE = True
        return 0.0


def require_plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(config.RESULTS_DIR / ".mplconfig")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    return plt, sns, pd


def load_joined_records(dataset: str) -> List[Dict]:
    result_dir = dataset_result_dir(config.RESULTS_DIR, dataset)
    predictions = {str(row["sample_id"]): row for row in read_jsonl(result_dir / "predictions.jsonl")}
    correctness = {str(row["sample_id"]): row for row in read_jsonl(result_dir / "correctness.jsonl")}
    embeddings = load_pickle(result_dir / "embeddings.pkl")
    if not predictions:
        raise FileNotFoundError(f"No predictions for {dataset}")
    if not correctness:
        raise FileNotFoundError(f"No correctness labels for {dataset}")
    if not embeddings:
        raise FileNotFoundError(f"No embeddings for {dataset}")

    sample_ids = [str(x) for x in embeddings["sample_ids"]]
    pred_embeddings = embeddings["pred_embeddings"]
    true_embeddings = embeddings["true_embeddings"]
    if len(sample_ids) != len(predictions) or len(sample_ids) != len(correctness):
        raise ValueError(
            f"Cached embeddings for {dataset} contain {len(sample_ids)} samples, "
            f"but predictions contain {len(predictions)} and correctness labels contain {len(correctness)}. "
            "Run run_embeddings.py again so embeddings match the current inference output."
        )
    rows = []
    for idx, sample_id in enumerate(sample_ids):
        if sample_id not in predictions or sample_id not in correctness:
            continue
        pred = predictions[sample_id]
        corr = correctness[sample_id]
        cosine = cosine_similarity(pred_embeddings[idx], true_embeddings[idx])
        exact = normalized_exact_match(pred["raw_prediction"], pred["true_answer"])
        tf1 = token_f1(pred["raw_prediction"], pred["true_answer"])
        bleu = compute_bleu(pred["raw_prediction"], pred["true_answer"])
        rows.append(
            {
                "dataset": dataset,
                "dataset_type": pred["dataset_type"],
                "sample_id": sample_id,
                "question": pred["question"],
                "raw_prediction": pred["raw_prediction"],
                "normalized_prediction": pred["normalized_prediction"],
                "true_answer": pred["true_answer"],
                "merged_prediction_statement": pred["merged_prediction_statement"],
                "merged_true_statement": pred["merged_true_statement"],
                "prediction_length": pred.get("prediction_length", word_count(pred.get("raw_prediction", ""))),
                "reference_length": word_count(pred.get("true_answer", "")),
                "global_cosine": cosine,
                "exact_match": exact,
                "token_f1": tf1,
                "bleu": bleu,
                "hhem_raw_score": corr["hhem_raw_score"],
                "hhem_correctness_score": corr["hhem_correctness_score"],
                "correct_label": corr["correct_label"],
            }
        )
    return rows


def save_curves(dataset: str, rows: List[Dict], metric_name: str, score_key: str) -> None:
    import numpy as np
    from semantic_similarity_eval.utils.metrics import curve_points

    plt, _, _ = require_plotting()
    y_true = np.asarray([row["correct_label"] for row in rows], dtype=int)
    scores = np.asarray([row[score_key] for row in rows], dtype=float)
    points = curve_points(y_true, scores)
    if not points:
        return
    fig_dir = ensure_dir(config.FIGURES_DIR / dataset)

    plt.figure(figsize=(4.2, 3.4))
    plt.plot(points["fpr"], points["tpr"], label=metric_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{dataset}: ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve.png", dpi=config.ANALYSIS["figure_dpi"])
    plt.savefig(fig_dir / "roc_curve.pdf")
    plt.close()

    plt.figure(figsize=(4.2, 3.4))
    plt.plot(points["recall"], points["precision"], label=metric_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{dataset}: Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "pr_curve.png", dpi=config.ANALYSIS["figure_dpi"])
    plt.savefig(fig_dir / "pr_curve.pdf")
    plt.close()


def threshold_predictor_analysis(dataset: str, rows: List[Dict]) -> None:
    import numpy as np
    from semantic_similarity_eval.utils.metrics import make_thresholds, threshold_sweep

    result_dir = dataset_result_dir(config.RESULTS_DIR, dataset)
    logger = setup_logging("threshold_analysis", result_dir / "threshold_analysis.log")

    y_true = np.asarray([row["correct_label"] for row in rows], dtype=int)

    # Analyze global cosine as threshold-based predictor
    scores = np.asarray([row["global_cosine"] for row in rows], dtype=float)
    sweep_results = threshold_sweep(y_true, scores, thresholds=make_thresholds(0.0, 1.0, 0.01))

    # Find best threshold
    best_f1_idx = np.argmax([r["f1"] for r in sweep_results])
    best_threshold = sweep_results[best_f1_idx]["threshold"]
    best_metrics = sweep_results[best_f1_idx]

    logger.info(f"Best threshold for {dataset}: {best_threshold:.2f}, F1: {best_metrics['f1']:.3f}, Precision: {best_metrics['precision']:.3f}, Recall: {best_metrics['recall']:.3f}")

    # Save threshold sweep results
    records_to_csv(
        result_dir / "threshold_predictor_analysis.csv",
        [
            {
                "dataset": dataset,
                "threshold": r["threshold"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "auc": r.get("auc", 0.0),
            }
            for r in sweep_results
        ]
    )


def save_dataset_plots(dataset: str, rows: List[Dict]) -> None:
    plt, sns, pd = require_plotting()
    frame = pd.DataFrame(rows)
    fig_dir = ensure_dir(config.FIGURES_DIR / dataset)

    plt.figure(figsize=(5.0, 3.4))
    sns.histplot(
        data=frame,
        x="global_cosine",
        hue="correct_label",
        bins=30,
        stat="density",
        common_norm=False,
        element="step",
    )
    plt.xlabel("Global Statement Cosine")
    plt.ylabel("Density")
    plt.title(f"{dataset}: Similarity Distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "similarity_distribution.png", dpi=config.ANALYSIS["figure_dpi"])
    plt.savefig(fig_dir / "similarity_distribution.pdf")
    plt.close()

    plt.figure(figsize=(4.6, 3.4))
    sns.violinplot(data=frame, x="correct_label", y="global_cosine", inner="box", cut=0)
    plt.xlabel("HHEM Correct Label")
    plt.ylabel("Global Statement Cosine")
    plt.title(f"{dataset}: Cosine by Correctness")
    plt.tight_layout()
    plt.savefig(fig_dir / "similarity_violin.png", dpi=config.ANALYSIS["figure_dpi"])
    plt.savefig(fig_dir / "similarity_violin.pdf")
    plt.close()

    plt.figure(figsize=(4.8, 3.4))
    sns.regplot(
        data=frame,
        x="prediction_length",
        y="global_cosine",
        scatter_kws={"alpha": 0.45, "s": 18},
        line_kws={"color": "black", "linewidth": 1.3},
    )
    plt.xlabel("Prediction Length (words)")
    plt.ylabel("Global Statement Cosine")
    plt.title(f"{dataset}: Length vs Similarity")
    plt.tight_layout()
    plt.savefig(fig_dir / "length_vs_similarity.png", dpi=config.ANALYSIS["figure_dpi"])
    plt.savefig(fig_dir / "length_vs_similarity.pdf")
    plt.close()

    plt.figure(figsize=(4.8, 3.4))
    sns.regplot(
        data=frame,
        x="hhem_correctness_score",
        y="global_cosine",
        scatter_kws={"alpha": 0.45, "s": 18},
        line_kws={"color": "black", "linewidth": 1.3},
    )
    plt.xlabel("HHEM Correctness Score")
    plt.ylabel("Global Statement Cosine")
    plt.title(f"{dataset}: HHEM vs Cosine")
    plt.tight_layout()
    plt.savefig(fig_dir / "hhem_vs_cosine.png", dpi=config.ANALYSIS["figure_dpi"])
    plt.savefig(fig_dir / "hhem_vs_cosine.pdf")
    plt.close()

    save_curves(dataset, rows, "Global cosine", "global_cosine")


def analyze_dataset(dataset: str) -> Dict:
    import numpy as np
    from semantic_similarity_eval.utils.metrics import make_thresholds, metric_summary, safe_corr, threshold_sweep

    result_dir = dataset_result_dir(config.RESULTS_DIR, dataset)
    logger = setup_logging("analyze_similarity", result_dir / "similarity.log")
    rows = load_joined_records(dataset)
    thresholds = make_thresholds(
        config.ANALYSIS["threshold_min"],
        config.ANALYSIS["threshold_max"],
        config.ANALYSIS["threshold_step"],
    )
    y_true = [row["correct_label"] for row in rows]
    scores = [row["global_cosine"] for row in rows]
    sweep = threshold_sweep(y_true, scores, thresholds)
    summary = metric_summary(y_true, scores, thresholds)
    corr = safe_corr([row["hhem_correctness_score"] for row in rows], scores)

    records_to_csv(result_dir / "similarity_scores.csv", rows)
    records_to_csv(result_dir / "threshold_sweep_global_cosine.csv", sweep)
    write_json(result_dir / "threshold_sweep_global_cosine.json", sweep)
    write_json(
        result_dir / "similarity_summary.json",
        {
            "dataset": dataset,
            "dataset_type": config.DATASETS[dataset]["type"],
            "n": len(rows),
            "positive_rate": float(np.mean(y_true)) if y_true else float("nan"),
            "global_cosine": summary,
            "hhem_cosine_correlation": corr,
        },
    )
    save_dataset_plots(dataset, rows)
    threshold_predictor_analysis(dataset, rows)
    logger.info("Analyzed %d records for %s", len(rows), dataset)
    return {
        "dataset": dataset,
        "dataset_type": config.DATASETS[dataset]["type"],
        "n": len(rows),
        "positive_rate": float(np.mean(y_true)) if y_true else float("nan"),
        "metric": "global_cosine",
        "best_threshold": summary["threshold"],
        "accuracy": summary["accuracy"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "best_f1": summary["f1"],
        "auroc": summary["auroc"],
        "auprc": summary["auprc"],
        "pearson_hhem_cosine": corr["pearson"],
        "spearman_hhem_cosine": corr["spearman"],
    }


def save_cross_dataset_plots(summary_rows: List[Dict], all_rows: List[Dict]) -> None:
    plt, sns, pd = require_plotting()
    ensure_dir(config.FIGURES_DIR)

    metric_rows = []
    for row in summary_rows:
        for metric in ["auroc", "auprc", "best_f1"]:
            value = row.get(metric)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                metric_rows.append({"dataset": row["dataset"], "metric": metric.upper(), "value": value})
    if metric_rows:
        plt.figure(figsize=(6.2, 3.6))
        sns.barplot(data=pd.DataFrame(metric_rows), x="dataset", y="value", hue="metric")
        plt.ylim(0, 1)
        plt.xlabel("Dataset")
        plt.ylabel("Score")
        plt.title("Global Cosine Performance by Dataset")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "dataset_metric_bars.png", dpi=config.ANALYSIS["figure_dpi"])
        plt.savefig(config.FIGURES_DIR / "dataset_metric_bars.pdf")
        plt.close()

    if all_rows:
        plt.figure(figsize=(6.0, 3.6))
        sns.boxplot(data=pd.DataFrame(all_rows), x="dataset_type", y="global_cosine", hue="correct_label")
        plt.xlabel("Dataset Type")
        plt.ylabel("Global Statement Cosine")
        plt.title("Short-form vs Long-form Similarity")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "short_vs_long_similarity_box.png", dpi=config.ANALYSIS["figure_dpi"])
        plt.savefig(config.FIGURES_DIR / "short_vs_long_similarity_box.pdf")
        plt.close()

        type_summary = short_vs_long_summary(all_rows)
        type_metric_rows = []
        for row in type_summary:
            for metric in ["auroc", "auprc", "best_f1"]:
                value = row.get(metric)
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    type_metric_rows.append({"dataset_type": row["dataset_type"], "metric": metric.upper(), "value": value})
        if type_metric_rows:
            plt.figure(figsize=(4.8, 3.4))
            sns.barplot(data=pd.DataFrame(type_metric_rows), x="dataset_type", y="value", hue="metric")
            plt.ylim(0, 1)
            plt.xlabel("Dataset Type")
            plt.ylabel("Score")
            plt.title("Short-form vs Long-form Metrics")
            plt.tight_layout()
            plt.savefig(config.FIGURES_DIR / "short_vs_long_metric_bars.png", dpi=config.ANALYSIS["figure_dpi"])
            plt.savefig(config.FIGURES_DIR / "short_vs_long_metric_bars.pdf")
            plt.close()


def short_vs_long_summary(all_rows: List[Dict]) -> List[Dict]:
    import numpy as np
    from semantic_similarity_eval.utils.metrics import make_thresholds, metric_summary

    thresholds = make_thresholds(
        config.ANALYSIS["threshold_min"],
        config.ANALYSIS["threshold_max"],
        config.ANALYSIS["threshold_step"],
    )
    summaries = []
    for dataset_type in ["short", "long"]:
        rows = [row for row in all_rows if row["dataset_type"] == dataset_type]
        if not rows:
            continue
        y_true = [row["correct_label"] for row in rows]
        scores = [row["global_cosine"] for row in rows]
        summary = metric_summary(y_true, scores, thresholds)
        summaries.append(
            {
                "dataset_type": dataset_type,
                "n": len(rows),
                "positive_rate": float(np.mean(y_true)),
                "best_threshold": summary["threshold"],
                "accuracy": summary["accuracy"],
                "precision": summary["precision"],
                "recall": summary["recall"],
                "best_f1": summary["f1"],
                "auroc": summary["auroc"],
                "auprc": summary["auprc"],
            }
        )
    return summaries


def main():
    args = parse_args()
    logger = setup_logging("analyze_similarity", config.RESULTS_DIR / "analyze_similarity.log")
    summary_rows = []
    all_rows = []
    for dataset in dataset_names(args.dataset):
        summary_rows.append(analyze_dataset(dataset))
        all_rows.extend(load_joined_records(dataset))

    if args.dataset is None:
        records_to_csv(config.RESULTS_DIR / "summary_metrics.csv", summary_rows)
        write_json(config.RESULTS_DIR / "summary_metrics.json", summary_rows)
        type_summary = short_vs_long_summary(all_rows)
        records_to_csv(config.RESULTS_DIR / "short_vs_long_summary.csv", type_summary)
        write_json(config.RESULTS_DIR / "short_vs_long_summary.json", type_summary)
        save_cross_dataset_plots(summary_rows, all_rows)
        logger.info("Saved global summary tables and figures")


if __name__ == "__main__":
    main()
