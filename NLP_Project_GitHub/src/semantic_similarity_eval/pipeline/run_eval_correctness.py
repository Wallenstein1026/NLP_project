import argparse
from typing import List, Optional

from tqdm import tqdm

from semantic_similarity_eval import config
from semantic_similarity_eval.utils.io import completed_sample_ids, dataset_result_dir, records_to_csv, read_jsonl, write_json, write_jsonl
from semantic_similarity_eval.utils.logging_utils import setup_logging
from semantic_similarity_eval.utils.modeling import clear_cuda_cache, get_device, load_hhem_model, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Score prediction statements with HHEM.")
    parser.add_argument("--dataset", choices=list(config.DATASETS), default=None)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dataset_names(dataset: Optional[str]) -> List[str]:
    return [dataset] if dataset else list(config.DATASETS.keys())


def detect_hhem_direction(model, logger) -> dict:
    import numpy as np

    pairs = [(row["premise"], row["hypothesis"]) for row in config.HHEM["sanity_pairs"]]
    labels = np.asarray([row["label"] for row in config.HHEM["sanity_pairs"]], dtype=np.int32)
    scores = model.predict(pairs).detach().cpu().numpy()
    pos_mean = float(scores[labels == 1].mean())
    neg_mean = float(scores[labels == 0].mean())
    high_means_correct = pos_mean >= neg_mean
    logger.info("HHEM sanity scores pos_mean=%.4f neg_mean=%.4f high_means_correct=%s", pos_mean, neg_mean, high_means_correct)
    return {
        "positive_mean": pos_mean,
        "negative_mean": neg_mean,
        "high_means_correct": high_means_correct,
        "scores": [float(score) for score in scores],
    }


def run_for_dataset(dataset: str, model, direction: dict, overwrite: bool = False) -> None:
    result_dir = dataset_result_dir(config.RESULTS_DIR, dataset)
    logger = setup_logging("run_eval_correctness", result_dir / "correctness.log")
    predictions_path = result_dir / "predictions.jsonl"
    output_path = result_dir / "correctness.jsonl"
    csv_path = result_dir / "correctness.csv"
    metadata_path = result_dir / "correctness_metadata.json"

    predictions = read_jsonl(predictions_path)
    if not predictions:
        raise FileNotFoundError(f"No predictions found at {predictions_path}. Run inference first.")
    if overwrite and output_path.exists():
        output_path.unlink()
    done_ids = set() if overwrite else completed_sample_ids(output_path)
    pending = [row for row in predictions if str(row["sample_id"]) not in done_ids]
    logger.info("Dataset=%s predictions=%d pending=%d", dataset, len(predictions), len(pending))

    pairs = [
        (row["merged_true_statement"], row["merged_prediction_statement"])
        for row in pending
    ]
    scores = []
    if pairs:
        for start in tqdm(range(0, len(pairs), config.HHEM["batch_size"]), desc=f"hhem:{dataset}"):
            batch = pairs[start:start + config.HHEM["batch_size"]]
            batch_scores = model.predict(batch).detach().cpu().numpy().tolist()
            scores.extend(float(score) for score in batch_scores)

    new_rows = []
    for row, raw_score in zip(pending, scores):
        correct_score = raw_score if direction["high_means_correct"] else 1.0 - raw_score
        new_rows.append(
            {
                "dataset": dataset,
                "dataset_type": row["dataset_type"],
                "sample_id": row["sample_id"],
                "hhem_raw_score": float(raw_score),
                "hhem_correctness_score": float(correct_score),
                "correct_label": int(correct_score >= config.HHEM["threshold"]),
                "hhem_threshold": config.HHEM["threshold"],
                "question": row["question"],
                "raw_prediction": row["raw_prediction"],
                "true_answer": row["true_answer"],
                "merged_prediction_statement": row["merged_prediction_statement"],
                "merged_true_statement": row["merged_true_statement"],
            }
        )
    if new_rows:
        write_jsonl(output_path, new_rows, append=True)

    final_rows = read_jsonl(output_path)
    records_to_csv(csv_path, final_rows)
    write_json(
        metadata_path,
        {
            "dataset": dataset,
            "hhem_model": str(config.MODEL_PATHS["hhem_model"]),
            "hhem_foundation_model": str(config.MODEL_PATHS["hhem_foundation_model"]),
            "threshold": config.HHEM["threshold"],
            "direction": direction,
            "completed_samples": len(final_rows),
        },
    )
    logger.info("Saved %d correctness records to %s", len(final_rows), output_path)


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = get_device(args.device)
    logger = setup_logging("run_eval_correctness", config.RESULTS_DIR / "run_eval_correctness.log")
    logger.info("Loading HHEM model from %s on %s", config.MODEL_PATHS["hhem_model"], device)
    model = load_hhem_model(
        config.MODEL_PATHS["hhem_model"],
        config.MODEL_PATHS["hhem_foundation_model"],
        device,
    )
    direction = detect_hhem_direction(model, logger)
    for dataset in dataset_names(args.dataset):
        run_for_dataset(dataset, model, direction, overwrite=args.overwrite)
    clear_cuda_cache()


if __name__ == "__main__":
    main()
