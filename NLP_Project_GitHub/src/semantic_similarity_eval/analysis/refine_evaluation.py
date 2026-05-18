import argparse
import random
from typing import Dict, List, Optional

from tqdm import tqdm

from semantic_similarity_eval import config
from semantic_similarity_eval.utils.io import dataset_result_dir, records_to_csv, read_jsonl, write_json
from semantic_similarity_eval.utils.logging_utils import setup_logging
from semantic_similarity_eval.utils.metrics import cosine_similarity, make_thresholds, metric_summary
from semantic_similarity_eval.utils.modeling import (
    clear_cuda_cache,
    encode_texts,
    get_device,
    load_embedding_model,
    load_hhem_model,
    set_seed,
)
from semantic_similarity_eval.utils.text_normalize import (
    clean_generated_statement,
    fallback_statement,
    list_set_match,
    normalized_exact_match,
    split_sentences,
    token_f1,
    word_count,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build corrected labels and compressed references for refined evaluation.")
    parser.add_argument("--dataset", choices=list(config.DATASETS), default=None)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dataset_names(dataset: Optional[str]) -> List[str]:
    return [dataset] if dataset else list(config.DATASETS.keys())


def hhem_direction_from_metadata(dataset: str) -> Dict:
    metadata_path = dataset_result_dir(config.RESULTS_DIR, dataset) / "correctness_metadata.json"
    import json

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata["direction"]


def source_records(dataset: str) -> List[Dict]:
    result_dir = dataset_result_dir(config.RESULTS_DIR, dataset)
    predictions = {str(row["sample_id"]): row for row in read_jsonl(result_dir / "predictions.jsonl")}
    correctness = {str(row["sample_id"]): row for row in read_jsonl(result_dir / "correctness.jsonl")}
    from semantic_similarity_eval.analysis.analyze_similarity import load_joined_records

    joined = {str(row["sample_id"]): row for row in load_joined_records(dataset)}
    rows = []
    for sample_id, pred in predictions.items():
        if sample_id not in correctness:
            continue
        corr = correctness[sample_id]
        joined_row = joined.get(sample_id, {})
        rows.append(
            {
                **pred,
                "global_cosine": joined_row.get("global_cosine", 0.0),
                "hhem_raw_score": corr["hhem_raw_score"],
                "hhem_correctness_score": corr["hhem_correctness_score"],
                "hhem_correct_label": corr.get("correct_label", corr.get("hhem_correct_label", 0)),
                "hhem_threshold": corr.get("hhem_threshold", config.HHEM["threshold"]),
            }
        )
    return rows


def reference_chunks(row: Dict) -> List[str]:
    candidates = split_sentences(row.get("merged_true_statement", ""))
    raw_chunks = split_sentences(row.get("true_answer", ""))
    chunks = []
    seen = set()
    for chunk in [*candidates, *raw_chunks]:
        text, _ = clean_generated_statement(chunk)
        text = str(text or "").strip()
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            chunks.append(text)
            seen.add(key)
    return chunks or [fallback_statement(row.get("question", ""), row.get("true_answer", ""))]


def select_reference_core(row: Dict, tokenizer, model, device: str) -> Dict:
    dataset = row["dataset"]
    dataset_type = row["dataset_type"]
    cleaned_true, true_truncated = clean_generated_statement(
        row.get("merged_true_statement", ""),
        fallback_statement(row.get("question", ""), row.get("true_answer", "")),
    )
    if dataset_type != "long" or dataset != "nq":
        return {
            "reference_answer_core": cleaned_true,
            "reference_core_source": "cleaned_statement",
            "reference_core_chunk_count": 1,
            "reference_core_word_count": word_count(cleaned_true),
            "true_statement_truncated": true_truncated,
        }

    chunks = reference_chunks({**row, "merged_true_statement": cleaned_true})
    query = str(row.get("question", "")).strip()
    texts = [query, *chunks]
    vectors = encode_texts(
        tokenizer,
        model,
        texts,
        device,
        config.EMBEDDING["batch_size"],
        config.EMBEDDING["max_length"],
    )
    query_vec = vectors[0]
    chunk_vecs = vectors[1:]
    scored = [
        (idx, cosine_similarity(query_vec, chunk_vecs[idx]), chunks[idx])
        for idx in range(len(chunks))
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    selected = sorted(scored[: min(2, len(scored))], key=lambda item: item[0])
    core = " ".join(item[2] for item in selected).strip() or cleaned_true
    return {
        "reference_answer_core": core,
        "reference_core_source": "embedding_top_chunks",
        "reference_core_chunk_count": len(selected),
        "reference_core_word_count": word_count(core),
        "true_statement_truncated": true_truncated,
    }


def label_from_rules(row: Dict, refined_hhem_label: int) -> Dict:
    exact = normalized_exact_match(row.get("raw_prediction", ""), row.get("true_answer", ""))
    tf1 = token_f1(row.get("raw_prediction", ""), row.get("true_answer", ""))
    list_match = list_set_match(row.get("raw_prediction", ""), row.get("true_answer", ""))

    if exact == 1.0:
        return {"final_correct_label": 1, "label_source": "exact_match_override"}
    if list_match == 1.0:
        return {"final_correct_label": 1, "label_source": "list_set_override"}
    if tf1 >= 0.95:
        return {"final_correct_label": 1, "label_source": "token_f1_override"}
    return {"final_correct_label": refined_hhem_label, "label_source": "hhem"}


def evaluate_rows(rows: List[Dict], label_key: str, metric_keys: List[str]) -> List[Dict]:
    thresholds = make_thresholds(
        config.ANALYSIS["threshold_min"],
        config.ANALYSIS["threshold_max"],
        config.ANALYSIS["threshold_step"],
    )
    comparisons = []
    for key in metric_keys:
        summary = metric_summary(
            [int(row[label_key]) for row in rows],
            [float(row[key]) for row in rows],
            thresholds,
        )
        comparisons.append(
            {
                "dataset": rows[0]["dataset"] if rows else "",
                "dataset_type": rows[0]["dataset_type"] if rows else "",
                "n": len(rows),
                "label_key": label_key,
                "metric": key,
                "best_threshold": summary["threshold"],
                "accuracy": summary["accuracy"],
                "precision": summary["precision"],
                "recall": summary["recall"],
                "best_f1": summary["f1"],
                "auroc": summary["auroc"],
                "auprc": summary["auprc"],
            }
        )
    return comparisons


def summary_row(rows: List[Dict], label_key: str, score_key: str = "refined_global_cosine") -> Dict:
    import numpy as np

    thresholds = make_thresholds(
        config.ANALYSIS["threshold_min"],
        config.ANALYSIS["threshold_max"],
        config.ANALYSIS["threshold_step"],
    )
    y_true = [int(row[label_key]) for row in rows]
    scores = [float(row[score_key]) for row in rows]
    summary = metric_summary(y_true, scores, thresholds)
    return {
        "dataset": rows[0]["dataset"] if rows else "",
        "dataset_type": rows[0]["dataset_type"] if rows else "",
        "n": len(rows),
        "positive_rate": float(np.mean(y_true)) if y_true else float("nan"),
        "label_key": label_key,
        "metric": score_key,
        "best_threshold": summary["threshold"],
        "accuracy": summary["accuracy"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "best_f1": summary["f1"],
        "auroc": summary["auroc"],
        "auprc": summary["auprc"],
    }


def score_hhem(pairs, hhem_model) -> List[float]:
    scores = []
    for start in tqdm(range(0, len(pairs), config.HHEM["batch_size"]), desc="refined_hhem"):
        batch = pairs[start:start + config.HHEM["batch_size"]]
        batch_scores = hhem_model.predict(batch).detach().cpu().numpy().tolist()
        scores.extend(float(score) for score in batch_scores)
    return scores


def run_for_dataset(dataset: str, tokenizer, embedding_model, hhem_model, device: str, overwrite: bool = False) -> Dict:
    result_dir = dataset_result_dir(config.RESULTS_DIR, dataset)
    logger = setup_logging("refine_evaluation", result_dir / "refine_evaluation.log")
    refined_scores_path = result_dir / "refined_similarity_scores.csv"
    refined_comparison_path = result_dir / "refined_metric_comparison.csv"
    metadata_path = result_dir / "refined_metadata.json"
    if refined_scores_path.exists() and refined_comparison_path.exists() and not overwrite:
        logger.info("Refined evaluation already exists for %s; use --overwrite to recompute.", dataset)
        from semantic_similarity_eval.utils.io import read_csv

        return {
            "rows": read_csv(refined_scores_path),
            "summary": {},
            "comparison": read_csv(refined_comparison_path),
            "short_audit": [],
            "long_audit": [],
        }

    rows = source_records(dataset)
    direction = hhem_direction_from_metadata(dataset)
    refined_rows = []
    short_audit = []
    long_audit = []

    logger.info("Building refined references for %s (%d rows)", dataset, len(rows))
    for row in tqdm(rows, desc=f"refine_reference:{dataset}"):
        pred_clean, pred_truncated = clean_generated_statement(
            row.get("merged_prediction_statement", ""),
            fallback_statement(row.get("question", ""), row.get("raw_prediction", "")),
        )
        core = select_reference_core(row, tokenizer, embedding_model, device)
        refined_rows.append(
            {
                **row,
                "cleaned_prediction_statement": pred_clean,
                "prediction_statement_truncated": pred_truncated,
                **core,
            }
        )

    pred_vectors = encode_texts(
        tokenizer,
        embedding_model,
        [row["cleaned_prediction_statement"] for row in refined_rows],
        device,
        config.EMBEDDING["batch_size"],
        config.EMBEDDING["max_length"],
    )
    ref_vectors = encode_texts(
        tokenizer,
        embedding_model,
        [row["reference_answer_core"] for row in refined_rows],
        device,
        config.EMBEDDING["batch_size"],
        config.EMBEDDING["max_length"],
    )
    refined_cosines = [
        cosine_similarity(pred_vectors[idx], ref_vectors[idx])
        for idx in range(len(refined_rows))
    ]

    hhem_pairs = [
        (row["reference_answer_core"], row["cleaned_prediction_statement"])
        for row in refined_rows
    ]
    refined_raw_scores = score_hhem(hhem_pairs, hhem_model)

    for idx, row in enumerate(refined_rows):
        refined_score = refined_raw_scores[idx] if direction["high_means_correct"] else 1.0 - refined_raw_scores[idx]
        refined_hhem_label = int(refined_score >= config.HHEM["threshold"])
        label = label_from_rules(row, refined_hhem_label)
        exact = normalized_exact_match(row.get("raw_prediction", ""), row.get("true_answer", ""))
        tf1 = token_f1(row.get("raw_prediction", ""), row.get("true_answer", ""))
        list_match = list_set_match(row.get("raw_prediction", ""), row.get("true_answer", ""))
        row.update(
            {
                "original_correct_label": row["hhem_correct_label"],
                "refined_global_cosine": refined_cosines[idx],
                "refined_hhem_raw_score": refined_raw_scores[idx],
                "refined_hhem_correctness_score": refined_score,
                "refined_hhem_correct_label": refined_hhem_label,
                "normalized_exact_match": exact,
                "token_f1": tf1,
                "list_set_match": list_match,
                **label,
            }
        )
        if label["label_source"] != "hhem":
            short_audit.append(
                {
                    "dataset": dataset,
                    "dataset_type": row["dataset_type"],
                    "sample_id": row["sample_id"],
                    "label_source": label["label_source"],
                    "original_correct_label": row["original_correct_label"],
                    "final_correct_label": row["final_correct_label"],
                    "hhem_correctness_score": row["hhem_correctness_score"],
                    "question": row["question"],
                    "prediction": row["raw_prediction"],
                    "reference": row["true_answer"],
                    "normalized_exact_match": exact,
                    "token_f1": tf1,
                    "list_set_match": list_match,
                }
            )
        if dataset == "nq":
            long_audit.append(
                {
                    "dataset": dataset,
                    "sample_id": row["sample_id"],
                    "question": row["question"],
                    "prediction": row["raw_prediction"],
                    "original_reference": row["true_answer"],
                    "reference_answer_core": row["reference_answer_core"],
                    "original_reference_length": word_count(row["true_answer"]),
                    "reference_core_word_count": row["reference_core_word_count"],
                    "reference_core_source": row["reference_core_source"],
                    "refined_hhem_correctness_score": row["refined_hhem_correctness_score"],
                    "final_correct_label": row["final_correct_label"],
                }
            )

    comparisons = evaluate_rows(
        refined_rows,
        "final_correct_label",
        ["global_cosine", "refined_global_cosine", "hhem_correctness_score", "refined_hhem_correctness_score"],
    )
    records_to_csv(refined_scores_path, refined_rows)
    records_to_csv(refined_comparison_path, comparisons)
    write_json(
        metadata_path,
        {
            "dataset": dataset,
            "completed_samples": len(refined_rows),
            "short_label_overrides": len(short_audit),
            "long_reference_core_audit_rows": len(long_audit),
            "statement_truncations": sum(
                1
                for row in refined_rows
                if row["prediction_statement_truncated"] or row["true_statement_truncated"]
            ),
            "reference_core_strategy": "nq_question_only_embedding_top_2_chunks_else_cleaned_statement",
        },
    )
    logger.info("Saved refined evaluation for %s", dataset)
    return {
        "rows": refined_rows,
        "summary": summary_row(refined_rows, "final_correct_label"),
        "comparison": comparisons,
        "short_audit": short_audit,
        "long_audit": long_audit,
    }


def save_global_outputs(results: List[Dict]) -> None:
    all_comparisons = []
    all_summaries = []
    all_rows = []
    all_short_audit = []
    all_long_audit = []
    for result in results:
        all_rows.extend(result["rows"])
        if result["summary"]:
            all_summaries.append(result["summary"])
        all_comparisons.extend(result["comparison"])
        all_short_audit.extend(result["short_audit"])
        all_long_audit.extend(result["long_audit"])
    records_to_csv(config.RESULTS_DIR / "refined_summary_metrics.csv", all_summaries)
    write_json(config.RESULTS_DIR / "refined_summary_metrics.json", all_summaries)
    records_to_csv(config.RESULTS_DIR / "refined_metric_comparison_all.csv", all_comparisons)
    write_json(config.RESULTS_DIR / "refined_metric_comparison_all.json", all_comparisons)
    records_to_csv(config.RESULTS_DIR / "label_audit_short.csv", all_short_audit)
    records_to_csv(config.RESULTS_DIR / "reference_core_audit_long.csv", all_long_audit)
    rng = random.Random(config.SEED)
    manual_review = list(all_long_audit)
    rng.shuffle(manual_review)
    records_to_csv(config.RESULTS_DIR / "reference_core_manual_review_50.csv", manual_review[:50])
    type_summaries = []
    for dataset_type in ["short", "long"]:
        rows = [row for row in all_rows if row["dataset_type"] == dataset_type]
        if rows:
            type_summaries.append(summary_row(rows, "final_correct_label"))
            type_summaries[-1]["dataset"] = dataset_type
            type_summaries[-1]["dataset_type"] = dataset_type
    records_to_csv(config.RESULTS_DIR / "refined_short_vs_long_summary.csv", type_summaries)
    write_json(config.RESULTS_DIR / "refined_short_vs_long_summary.json", type_summaries)


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = get_device(args.device)
    logger = setup_logging("refine_evaluation", config.RESULTS_DIR / "refine_evaluation.log")
    logger.info("Loading embedding model from %s on %s", config.MODEL_PATHS["embedding_model"], device)
    tokenizer, embedding_model = load_embedding_model(config.MODEL_PATHS["embedding_model"], device, config.TORCH_DTYPE)
    logger.info("Loading HHEM model from %s on %s", config.MODEL_PATHS["hhem_model"], device)
    hhem_model = load_hhem_model(
        config.MODEL_PATHS["hhem_model"],
        config.MODEL_PATHS["hhem_foundation_model"],
        device,
    )
    results = []
    for dataset in dataset_names(args.dataset):
        results.append(run_for_dataset(dataset, tokenizer, embedding_model, hhem_model, device, overwrite=args.overwrite))
    if args.dataset is None:
        save_global_outputs(results)
    clear_cuda_cache()


if __name__ == "__main__":
    main()
