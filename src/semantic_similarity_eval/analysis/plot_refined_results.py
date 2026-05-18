import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

from semantic_similarity_eval import config
from semantic_similarity_eval.utils.io import ensure_dir


SUMMARY_FILES = [
    "summary_metrics.csv",
    "metric_comparison_all.csv",
    "short_vs_long_summary.csv",
    "refined_summary_metrics.csv",
    "refined_metric_comparison_all.csv",
    "refined_short_vs_long_summary.csv",
    "label_audit_short.csv",
    "reference_core_audit_long.csv",
    "reference_core_manual_review_50.csv",
]

DATASET_ORDER = list(config.DATASETS.keys())
METRIC_ORDER = ["auroc", "auprc", "best_f1"]
METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "best_f1": "Best F1",
    "positive_rate": "Positive Rate",
}
STAGE_LABELS = {
    "original": "Original",
    "refined": "Refined",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate refined evaluation figures in results_refine.")
    parser.add_argument("--dataset", choices=DATASET_ORDER, default=None)
    parser.add_argument("--output-dir", type=Path, default=config.RESULTS_REFINE_DIR)
    return parser.parse_args()


def require_plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(config.RESULTS_REFINE_DIR / ".mplconfig")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    return plt, sns, pd


def savefig(fig, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=config.ANALYSIS["figure_dpi"])
    fig.savefig(path_without_suffix.with_suffix(".pdf"))


def copy_tables(output_dir: Path) -> None:
    table_dir = ensure_dir(output_dir / "tables")
    for filename in SUMMARY_FILES:
        src = config.RESULTS_DIR / filename
        dst = table_dir / filename
        if src.exists() and src.stat().st_size > 2:
            shutil.copy2(src, dst)


def read_required_csv(pd, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def read_csv_with_fallback(pd, primary: Path, fallback: Path):
    try:
        if primary.exists() and primary.stat().st_size > 2:
            return pd.read_csv(primary)
    except pd.errors.EmptyDataError:
        pass
    if fallback.exists() and fallback.stat().st_size > 2:
        return pd.read_csv(fallback)
    return read_required_csv(pd, primary)


def ordered_datasets(frame) -> List[str]:
    present = set(frame["dataset"].astype(str).tolist())
    return [dataset for dataset in DATASET_ORDER if dataset in present]


def build_before_after_frame(pd, original_summary, refined_summary):
    rows = []
    for stage, frame in [("original", original_summary), ("refined", refined_summary)]:
        for _, row in frame.iterrows():
            dataset = str(row["dataset"])
            rows.append(
                {
                    "dataset": dataset,
                    "stage": STAGE_LABELS[stage],
                    "metric": "Positive Rate",
                    "value": float(row["positive_rate"]),
                }
            )
            for key in METRIC_ORDER:
                rows.append(
                    {
                        "dataset": dataset,
                        "stage": STAGE_LABELS[stage],
                        "metric": METRIC_LABELS[key],
                        "value": float(row[key]),
                    }
                )
    return pd.DataFrame(rows)


def plot_before_after(summary_frame, fig_dir: Path) -> None:
    plt, sns, pd = require_plotting()
    datasets = ordered_datasets(summary_frame)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), sharey=True)
    for ax, metric in zip(axes, [METRIC_LABELS[key] for key in METRIC_ORDER]):
        subset = summary_frame[summary_frame["metric"] == metric]
        sns.barplot(data=subset, x="dataset", y="value", hue="stage", order=datasets, ax=ax)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Dataset")
        ax.set_ylabel(metric if ax is axes[0] else "")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
        if ax is not axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="")
    fig.suptitle("Original vs Refined: Ranking Metrics", y=1.02)
    fig.tight_layout()
    savefig(fig, fig_dir / "before_after_ranking_metrics")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    subset = summary_frame[summary_frame["metric"] == "Positive Rate"]
    sns.barplot(data=subset, x="dataset", y="value", hue="stage", order=datasets, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Positive Rate")
    ax.set_title("Original vs Refined: Positive Rate")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="")
    fig.tight_layout()
    savefig(fig, fig_dir / "before_after_positive_rate")
    plt.close(fig)


def plot_refined_summary(refined_summary, fig_dir: Path) -> None:
    plt, sns, pd = require_plotting()
    rows = []
    for _, row in refined_summary.iterrows():
        for key in METRIC_ORDER:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "metric": METRIC_LABELS[key],
                    "value": float(row[key]),
                }
            )
    plot_frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    sns.barplot(
        data=plot_frame,
        x="dataset",
        y="value",
        hue="metric",
        order=ordered_datasets(refined_summary),
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Score")
    ax.set_title("Refined Global Cosine Performance")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="")
    fig.tight_layout()
    savefig(fig, fig_dir / "refined_summary_metrics")
    plt.close(fig)


def plot_short_vs_long(short_long_frame, fig_dir: Path) -> None:
    plt, sns, pd = require_plotting()
    rows = []
    for _, row in short_long_frame.iterrows():
        for key in ["positive_rate", *METRIC_ORDER]:
            rows.append(
                {
                    "dataset_type": row["dataset_type"],
                    "metric": METRIC_LABELS[key],
                    "value": float(row[key]),
                }
            )
    plot_frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.4), sharey=True)
    for ax, metric in zip(axes, ["Positive Rate", "AUROC", "AUPRC", "Best F1"]):
        subset = plot_frame[plot_frame["metric"] == metric]
        sns.barplot(data=subset, x="dataset_type", y="value", order=["short", "long"], ax=ax)
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel(metric if ax is axes[0] else "")
        ax.set_title(metric)
    fig.suptitle("Refined Results by Answer Type", y=1.02)
    fig.tight_layout()
    savefig(fig, fig_dir / "refined_short_vs_long_metrics")
    plt.close(fig)


def plot_metric_comparison(refined_comparison, fig_dir: Path) -> None:
    plt, sns, pd = require_plotting()
    subset = refined_comparison[
        refined_comparison["metric"].isin(["global_cosine", "refined_global_cosine"])
    ].copy()
    subset["method"] = subset["metric"].map(
        {
            "global_cosine": "Original cosine + final label",
            "refined_global_cosine": "Refined cosine + final label",
        }
    )
    datasets = ordered_datasets(subset)
    for key in METRIC_ORDER:
        fig, ax = plt.subplots(figsize=(7.4, 3.7))
        sns.barplot(data=subset, x="dataset", y=key, hue="method", order=datasets, ax=ax)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Dataset")
        ax.set_ylabel(METRIC_LABELS[key])
        ax.set_title(f"Cosine Comparison under Final Labels: {METRIC_LABELS[key]}")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="")
        fig.tight_layout()
        savefig(fig, fig_dir / f"refined_metric_comparison_{key}")
        plt.close(fig)


def plot_dataset_distribution(dataset: str, score_frame, fig_dir: Path) -> None:
    plt, sns, pd = require_plotting()
    dataset_dir = ensure_dir(fig_dir / dataset)
    frame = score_frame.copy()
    frame["final_correct_label"] = frame["final_correct_label"].astype(int)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    sns.histplot(
        data=frame,
        x="refined_global_cosine",
        hue="final_correct_label",
        bins=30,
        stat="density",
        common_norm=False,
        element="step",
        ax=ax,
    )
    ax.set_xlabel("Refined Global Cosine")
    ax.set_ylabel("Density")
    ax.set_title(f"{dataset}: Refined Similarity Distribution")
    fig.tight_layout()
    savefig(fig, dataset_dir / "refined_similarity_distribution")
    plt.close(fig)

    if "global_cosine" in frame.columns:
        long_rows = []
        for _, row in frame.iterrows():
            long_rows.append(
                {
                    "label": int(row["final_correct_label"]),
                    "method": "Original cosine",
                    "value": float(row["global_cosine"]),
                }
            )
            long_rows.append(
                {
                    "label": int(row["final_correct_label"]),
                    "method": "Refined cosine",
                    "value": float(row["refined_global_cosine"]),
                }
            )
        long_frame = pd.DataFrame(long_rows)
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        sns.violinplot(data=long_frame, x="label", y="value", hue="method", cut=0, inner="box", ax=ax)
        ax.set_xlabel("Final Correct Label")
        ax.set_ylabel("Cosine")
        ax.set_title(f"{dataset}: Original vs Refined Cosine")
        ax.legend(title="")
        fig.tight_layout()
        savefig(fig, dataset_dir / "original_vs_refined_cosine_violin")
        plt.close(fig)


def plot_dataset_curves(dataset: str, score_frame, fig_dir: Path) -> None:
    import numpy as np
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    plt, _, _ = require_plotting()
    dataset_dir = ensure_dir(fig_dir / dataset)
    y_true = score_frame["final_correct_label"].astype(int).to_numpy()
    scores = score_frame["refined_global_cosine"].astype(float).to_numpy()
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, scores)
    auroc = roc_auc_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{dataset}: Refined ROC")
    ax.legend()
    fig.tight_layout()
    savefig(fig, dataset_dir / "refined_roc_curve")
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    auprc = average_precision_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{dataset}: Refined Precision-Recall")
    ax.legend()
    fig.tight_layout()
    savefig(fig, dataset_dir / "refined_pr_curve")
    plt.close(fig)



def export_failure_assets(pd, output_dir: Path) -> None:
    table_dir = ensure_dir(output_dir / "tables")
    frames = []
    for dataset in DATASET_ORDER:
        path = config.RESULTS_DIR / dataset / "failure_cases" / "all_failure_cases.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return
    failures = pd.concat(frames, ignore_index=True)
    failures.to_csv(table_dir / "failure_cases_all.csv", index=False)

    if "rank_score" in failures.columns:
        failures = failures.sort_values(["failure_type", "dataset", "rank_score"], ascending=[True, True, False])
    diversified = []
    for _, group in failures.groupby("failure_type"):
        first_per_dataset = group.groupby("dataset", group_keys=False).head(1)
        diversified.append(first_per_dataset.head(3))
    examples = pd.concat(diversified, ignore_index=True) if diversified else failures.head(0).copy()
    example_columns = [
        "failure_type",
        "dataset",
        "dataset_type",
        "sample_id",
        "question",
        "prediction",
        "reference",
        "correct_label",
        "global_cosine",
        "hhem_correctness_score",
        "token_f1",
        "rank_score",
    ]
    examples[[column for column in example_columns if column in examples.columns]].to_csv(
        table_dir / "failure_case_examples.csv",
        index=False,
    )


def export_embedding_ablation_assets(pd, output_dir: Path) -> None:
    table_dir = ensure_dir(output_dir / "tables")
    frames = []
    ablation_root = output_dir / "ablation"
    if not ablation_root.exists():
        return
    for path in sorted(ablation_root.glob("*/ablation_metric_comparison.csv")):
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if frames:
        summary = pd.concat(frames, ignore_index=True)
        summary.to_csv(table_dir / "component_ablation_summary.csv", index=False)
        summary.to_csv(table_dir / "embedding_ablation_summary.csv", index=False)


def export_refinement_ablation_assets(pd, output_dir: Path) -> None:
    table_dir = ensure_dir(output_dir / "tables")
    frames = []
    ablation_root = output_dir / "refinement_ablation"
    if not ablation_root.exists():
        return
    for path in sorted(ablation_root.glob("*/refinement_ablation_summary.csv")):
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(table_dir / "refinement_ablation_summary.csv", index=False)



def plot_embedding_ablation(output_dir: Path, fig_dir: Path) -> None:
    summary_path = output_dir / "tables" / "embedding_ablation_summary.csv"
    if not summary_path.exists() or summary_path.stat().st_size <= 2:
        return
    plt, sns, pd = require_plotting()
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return
    ablation_fig_dir = ensure_dir(fig_dir / "embedding_ablation")
    datasets = [dataset for dataset in DATASET_ORDER if dataset in set(summary["dataset"].astype(str))]
    if "method" not in summary.columns:
        return

    method_order = [
        method for method in [
            "global_cosine_only",
            "alignment_only",
            "token_f1_only",
            "exact_match_only",
            "semantic_components_only",
            "lexical_components_only",
            "all_components_uniform",
            "short_hybrid_full",
            "hybrid_without_exact_match",
            "hybrid_without_token_f1",
            "hybrid_without_cosine",
            "type_strategy_full",
        ]
        if method in set(summary["method"].astype(str))
    ]
    method_labels = {
        "global_cosine_only": "Cosine",
        "alignment_only": "Alignment",
        "token_f1_only": "Token F1",
        "exact_match_only": "Exact Match",
        "semantic_components_only": "Semantic Avg",
        "lexical_components_only": "Lexical Avg",
        "all_components_uniform": "Uniform All",
        "short_hybrid_full": "Short Hybrid",
        "hybrid_without_exact_match": "No Exact",
        "hybrid_without_token_f1": "No Token F1",
        "hybrid_without_cosine": "No Cosine",
        "type_strategy_full": "Type Strategy",
    }
    plot_summary = summary.copy()
    plot_summary["method_label"] = plot_summary["method"].map(method_labels).fillna(plot_summary["method"])

    for metric in ["best_f1", "auprc", "auroc"]:
        pivot = (
            plot_summary.pivot_table(index="method_label", columns="dataset", values=metric, aggfunc="mean")
            .reindex([method_labels[method] for method in method_order])
            .reindex(columns=datasets)
        )
        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        sns.heatmap(pivot, vmin=0, vmax=1, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5, ax=ax)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Ablation Method")
        ax.set_title(f"Component Ablation: {METRIC_LABELS[metric]}")
        fig.tight_layout()
        savefig(fig, ablation_fig_dir / f"component_ablation_{metric}_heatmap")
        plt.close(fig)

    best_rows = (
        plot_summary.sort_values(["dataset", "best_f1", "auprc", "auroc"], ascending=[True, False, False, False])
        .groupby("dataset", as_index=False)
        .head(1)
    )
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    sns.barplot(data=best_rows, x="dataset", y="best_f1", hue="method_label", order=datasets, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Best F1")
    ax.set_title("Component Ablation: Best Method per Dataset")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    savefig(fig, ablation_fig_dir / "component_ablation_best_method")
    plt.close(fig)

    threshold_pivot = (
        plot_summary.pivot_table(index="method_label", columns="dataset", values="best_threshold", aggfunc="mean")
        .reindex([method_labels[method] for method in method_order])
        .reindex(columns=datasets)
    )
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    sns.heatmap(threshold_pivot, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="mako", linewidths=0.5, ax=ax)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Ablation Method")
    ax.set_title("Component Ablation: Best Threshold")
    fig.tight_layout()
    savefig(fig, ablation_fig_dir / "component_ablation_best_threshold_heatmap")
    plt.close(fig)


def plot_refinement_ablation(output_dir: Path, fig_dir: Path) -> None:
    summary_path = output_dir / "tables" / "refinement_ablation_summary.csv"
    if not summary_path.exists() or summary_path.stat().st_size <= 2:
        return
    plt, sns, pd = require_plotting()
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return
    ablation_fig_dir = ensure_dir(fig_dir / "refinement_ablation")
    datasets = [dataset for dataset in DATASET_ORDER if dataset in set(summary["dataset"].astype(str))]
    variant_order = [
        "base",
        "semantic_refinement",
        "claim_chunk_alignment",
        "label_override_only",
        "all_refinements",
        "short_answer_hybrid",
        "short_hybrid_without_exact",
        "short_hybrid_without_token_f1",
    ]
    variant_labels = {
        "base": "Base",
        "semantic_refinement": "+ Semantic Refinement",
        "claim_chunk_alignment": "+ Claim/Chunk Alignment",
        "label_override_only": "+ Label Override",
        "all_refinements": "+ All Refinements",
        "short_answer_hybrid": "Short Hybrid",
        "short_hybrid_without_exact": "Short Hybrid - Exact",
        "short_hybrid_without_token_f1": "Short Hybrid - Token F1",
    }
    plot_summary = summary.copy()
    plot_summary["variant_label"] = plot_summary["variant"].map(variant_labels).fillna(plot_summary["variant"])

    long_rows = []
    for _, row in plot_summary.iterrows():
        for metric in ["best_f1", "auprc", "auroc"]:
            long_rows.append(
                {
                    "dataset": row["dataset"],
                    "variant": row["variant"],
                    "variant_label": row["variant_label"],
                    "metric": METRIC_LABELS[metric],
                    "value": float(row[metric]),
                }
            )
    metric_frame = pd.DataFrame(long_rows)
    for metric in ["Best F1", "AUPRC", "AUROC"]:
        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        subset = metric_frame[metric_frame["metric"] == metric]
        sns.barplot(
            data=subset,
            x="dataset",
            y="value",
            hue="variant_label",
            order=datasets,
            hue_order=[variant_labels[variant] for variant in variant_order],
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Dataset")
        ax.set_ylabel(metric)
        ax.set_title(f"Refinement Ablation: {metric}")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        savefig(fig, ablation_fig_dir / f"refinement_ablation_{metric.lower().replace(' ', '_')}")
        plt.close(fig)

    pivot = (
        plot_summary.pivot_table(index="variant_label", columns="dataset", values="best_f1", aggfunc="mean")
        .reindex([variant_labels[variant] for variant in variant_order])
        .reindex(columns=datasets)
    )
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    sns.heatmap(pivot, vmin=0, vmax=1, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5, ax=ax)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Pipeline Variant")
    ax.set_title("Refinement Ablation: Best F1")
    fig.tight_layout()
    savefig(fig, ablation_fig_dir / "refinement_ablation_best_f1_heatmap")
    plt.close(fig)

    threshold_pivot = (
        plot_summary.pivot_table(index="variant_label", columns="dataset", values="best_threshold", aggfunc="mean")
        .reindex([variant_labels[variant] for variant in variant_order])
        .reindex(columns=datasets)
    )
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    sns.heatmap(threshold_pivot, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="mako", linewidths=0.5, ax=ax)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Pipeline Variant")
    ax.set_title("Refinement Ablation: Best Threshold")
    fig.tight_layout()
    savefig(fig, ablation_fig_dir / "refinement_ablation_best_threshold_heatmap")
    plt.close(fig)


def plot_type_calibrated_ablation(output_dir: Path, fig_dir: Path) -> None:
    summary_path = output_dir / "tables" / "refinement_ablation_type_calibrated_summary.csv"
    if not summary_path.exists() or summary_path.stat().st_size <= 2:
        return
    plt, sns, pd = require_plotting()
    summary = pd.read_csv(summary_path)
    summary = summary[summary["calibration_scope"] == "dataset_type_pooled"].copy()
    if summary.empty:
        return
    ablation_fig_dir = ensure_dir(fig_dir / "refinement_ablation")
    variant_order = [
        "base",
        "semantic_refinement",
        "claim_chunk_alignment",
        "all_refinements",
        "short_answer_hybrid",
    ]
    variant_labels = {
        "base": "Base",
        "semantic_refinement": "+ Semantic Refinement",
        "claim_chunk_alignment": "+ Claim/Chunk Alignment",
        "all_refinements": "+ All Refinements",
        "short_answer_hybrid": "Short Hybrid",
    }
    summary = summary[summary["variant"].isin(variant_order)].copy()
    summary["variant_label"] = summary["variant"].map(variant_labels)
    hue_order = [variant_labels[variant] for variant in variant_order if variant in set(summary["variant"])]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    sns.barplot(data=summary, x="dataset_type", y="best_f1", hue="variant_label", order=["short", "long"], hue_order=hue_order, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Dataset Type")
    ax.set_ylabel("F1")
    ax.set_title("Type-Calibrated Threshold: Pooled F1")
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    savefig(fig, ablation_fig_dir / "type_calibrated_f1")
    plt.close(fig)

    pivot = (
        summary.pivot_table(index="variant_label", columns="dataset_type", values="calibrated_threshold", aggfunc="mean")
        .reindex(hue_order)
        .reindex(columns=["short", "long"])
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    sns.heatmap(pivot, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="mako", linewidths=0.5, ax=ax)
    ax.set_xlabel("Dataset Type")
    ax.set_ylabel("Variant")
    ax.set_title("Type-Calibrated Thresholds")
    fig.tight_layout()
    savefig(fig, ablation_fig_dir / "type_calibrated_thresholds")
    plt.close(fig)

def export_final_conclusion_tables(pd, output_dir: Path, original_summary, refined_summary, refined_short_long) -> None:
    table_dir = ensure_dir(output_dir / "tables")
    dataset_rows = []
    refined_by_dataset = {str(row["dataset"]): row for _, row in refined_summary.iterrows()}
    for _, base in original_summary.iterrows():
        dataset = str(base["dataset"])
        refined = refined_by_dataset.get(dataset)
        if refined is None:
            continue
        dataset_rows.append(
            {
                "scope": "dataset",
                "dataset": dataset,
                "dataset_type": base["dataset_type"],
                "n": int(base["n"]),
                "baseline_metric": "global_cosine + original_hhem_label",
                "refined_metric": "refined_global_cosine + final_correct_label",
                "baseline_auroc": float(base["auroc"]),
                "refined_auroc": float(refined["auroc"]),
                "delta_auroc": float(refined["auroc"]) - float(base["auroc"]),
                "baseline_auprc": float(base["auprc"]),
                "refined_auprc": float(refined["auprc"]),
                "delta_auprc": float(refined["auprc"]) - float(base["auprc"]),
                "baseline_best_f1": float(base["best_f1"]),
                "refined_best_f1": float(refined["best_f1"]),
                "delta_best_f1": float(refined["best_f1"]) - float(base["best_f1"]),
            }
        )

    short_long_path = config.RESULTS_DIR / "short_vs_long_summary.csv"
    if short_long_path.exists():
        baseline_short_long = pd.read_csv(short_long_path)
        refined_by_type = {str(row["dataset_type"]): row for _, row in refined_short_long.iterrows()}
        for _, base in baseline_short_long.iterrows():
            dataset_type = str(base["dataset_type"])
            refined = refined_by_type.get(dataset_type)
            if refined is None:
                continue
            dataset_rows.append(
                {
                    "scope": "answer_type",
                    "dataset": dataset_type,
                    "dataset_type": dataset_type,
                    "n": int(base["n"]),
                    "baseline_metric": "global_cosine + original_hhem_label",
                    "refined_metric": "refined_global_cosine + final_correct_label",
                    "baseline_auroc": float(base["auroc"]),
                    "refined_auroc": float(refined["auroc"]),
                    "delta_auroc": float(refined["auroc"]) - float(base["auroc"]),
                    "baseline_auprc": float(base["auprc"]),
                    "refined_auprc": float(refined["auprc"]),
                    "delta_auprc": float(refined["auprc"]) - float(base["auprc"]),
                    "baseline_best_f1": float(base["best_f1"]),
                    "refined_best_f1": float(refined["best_f1"]),
                    "delta_best_f1": float(refined["best_f1"]) - float(base["best_f1"]),
                }
            )
    pd.DataFrame(dataset_rows).to_csv(table_dir / "final_conclusion_table.csv", index=False)


def write_report_notes(output_dir: Path) -> None:
    content = """# Report Notes

## Baseline vs. Refined Results

The baseline experiment evaluates whether global cosine similarity between the original prediction statement and the original reference statement can predict the original HHEM correctness label. This is the direct test of embedding similarity as a correctness proxy.

The refined experiment evaluates refined global cosine against `final_correct_label`. The refined setting is designed to remove known artifacts before measuring semantic similarity, so it should be presented as an improvement method rather than as the same baseline condition.

## Refined Method

1. Statement cleaning: remove prompt/template leakage from generated standalone statements before embedding and entailment scoring.
2. Reference core extraction: for long-form references, select the answer-relevant reference core instead of embedding the full noisy reference passage.
3. Short-form lexical override: correct obvious short-answer label errors with normalized exact match, list-set match, and high token F1 rules.
4. Refined cosine: compute cosine similarity between cleaned prediction statements and refined reference statements, then evaluate threshold-based correctness prediction.

## Interpretation

Original global cosine is a weak correctness proxy on the current outputs. The refined pipeline improves AUROC, AUPRC, and Best F1, especially on short-form QA. Long-form QA remains harder because partial correctness, long references, and multi-fact answers make one global embedding score less reliable.

`refined_hhem_correctness_score` should not be used as the primary independent metric because `final_correct_label` partly depends on refined HHEM plus rule-based overrides. The main report should emphasize `global_cosine`, `refined_global_cosine`, threshold performance, and qualitative failure cases.

## Current Component Ablation Status

Ablation outputs belong under `outputs/results_refine/ablation/` and summaries under `outputs/results_refine/tables/`. The current ablation does not compare embedding model checkpoints; instead, it compares existing scoring components such as global cosine, lexical overlap, alignment, hybrid scores, and leave-one-component-out variants. Each row reports the best threshold selected from the configured sweep.
"""
    (output_dir / "report_notes.md").write_text(content, encoding="utf-8")

def write_readme(output_dir: Path) -> None:
    content = """# Refined Evaluation Figures

This directory is generated by `PYTHONPATH=src python -m semantic_similarity_eval plot`.

Main figures:
- `figures/before_after_ranking_metrics.png`: original vs refined AUROC, AUPRC, and Best F1.
- `figures/before_after_positive_rate.png`: original vs refined positive rate.
- `figures/refined_summary_metrics.png`: refined global cosine performance by dataset.
- `figures/refined_short_vs_long_metrics.png`: refined short-answer vs long-answer comparison.
- `figures/refined_metric_comparison_*.png`: original cosine vs refined cosine under `final_correct_label`.
- `figures/{dataset}/`: per-dataset refined distributions and ROC/PR curves.
- `figures/embedding_ablation/`: component ablation heatmaps and best-method visualizations.

Main tables:
- `tables/final_conclusion_table.csv`: baseline vs refined metrics by dataset and answer type.
- `tables/failure_cases_all.csv`: merged failure analysis cases across datasets.
- `tables/failure_case_examples.csv`: diversified qualitative examples, up to three per failure type.
- `tables/component_ablation_summary.csv`: component ablation metrics written from `outputs/results_refine/ablation/`.
- `tables/embedding_ablation_summary.csv`: compatibility copy of the component ablation summary.
- `report_notes.md`: report-ready text distinguishing baseline and refined methods.

Interpretation note: `refined_hhem_correctness_score` is not plotted as a primary method here because `final_correct_label` partly depends on HHEM after rule-based overrides. The main report figure should focus on `refined_global_cosine` and the before/after comparison.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def dataset_names(selected: Optional[str]) -> Iterable[str]:
    return [selected] if selected else DATASET_ORDER


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    fig_dir = ensure_dir(output_dir / "figures")

    plt, _, pd = require_plotting()
    original_summary = read_csv_with_fallback(
        pd,
        config.RESULTS_DIR / "summary_metrics.csv",
        output_dir / "tables" / "summary_metrics.csv",
    )
    refined_summary = read_csv_with_fallback(
        pd,
        config.RESULTS_DIR / "refined_summary_metrics.csv",
        output_dir / "tables" / "refined_summary_metrics.csv",
    )
    refined_comparison = read_csv_with_fallback(
        pd,
        config.RESULTS_DIR / "refined_metric_comparison_all.csv",
        output_dir / "tables" / "refined_metric_comparison_all.csv",
    )
    refined_short_long = read_csv_with_fallback(
        pd,
        config.RESULTS_DIR / "refined_short_vs_long_summary.csv",
        output_dir / "tables" / "refined_short_vs_long_summary.csv",
    )

    copy_tables(output_dir)
    export_failure_assets(pd, output_dir)
    export_refinement_ablation_assets(pd, output_dir)
    export_embedding_ablation_assets(pd, output_dir)
    export_final_conclusion_tables(pd, output_dir, original_summary, refined_summary, refined_short_long)
    write_report_notes(output_dir)
    before_after = build_before_after_frame(pd, original_summary, refined_summary)
    before_after.to_csv(output_dir / "tables" / "before_after_summary_long.csv", index=False)

    plot_before_after(before_after, fig_dir)
    plot_refined_summary(refined_summary, fig_dir)
    plot_short_vs_long(refined_short_long, fig_dir)
    plot_metric_comparison(refined_comparison, fig_dir)
    plot_refinement_ablation(output_dir, fig_dir)
    plot_type_calibrated_ablation(output_dir, fig_dir)
    plot_embedding_ablation(output_dir, fig_dir)

    for dataset in dataset_names(args.dataset):
        path = config.RESULTS_DIR / dataset / "refined_similarity_scores.csv"
        if not path.exists():
            continue
        scores = pd.read_csv(path)
        plot_dataset_distribution(dataset, scores, fig_dir)
        plot_dataset_curves(dataset, scores, fig_dir)

    write_readme(output_dir)
    plt.close("all")
    print(f"Saved refined figures to {fig_dir}")


if __name__ == "__main__":
    main()
