import argparse
from typing import List, Optional

from tqdm import tqdm

from semantic_similarity_eval import config
from semantic_similarity_eval.utils.io import dataset_result_dir, load_pickle, records_to_csv, read_jsonl, save_pickle, write_json
from semantic_similarity_eval.utils.logging_utils import setup_logging
from semantic_similarity_eval.utils.modeling import clear_cuda_cache, encode_texts, get_device, load_embedding_model, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Encode prediction and reference statements.")
    parser.add_argument("--dataset", choices=list(config.DATASETS), default=None)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dataset_names(dataset: Optional[str]) -> List[str]:
    return [dataset] if dataset else list(config.DATASETS.keys())


def embeddings_match_predictions(output_path, predictions: List[dict]) -> bool:
    embeddings = load_pickle(output_path, default=None)
    if not embeddings or "sample_ids" not in embeddings:
        return False
    cached_ids = [str(sample_id) for sample_id in embeddings["sample_ids"]]
    current_ids = [str(row["sample_id"]) for row in predictions]
    return cached_ids == current_ids


def run_for_dataset(dataset: str, tokenizer, model, device: str, overwrite: bool = False) -> None:
    from semantic_similarity_eval.utils.metrics import cosine_similarity

    result_dir = dataset_result_dir(config.RESULTS_DIR, dataset)
    logger = setup_logging("run_embeddings", result_dir / "embeddings.log")
    predictions_path = result_dir / "predictions.jsonl"
    output_path = result_dir / "embeddings.pkl"
    scores_path = result_dir / "embedding_cosine.csv"
    metadata_path = result_dir / "embedding_metadata.json"

    predictions = read_jsonl(predictions_path)
    if not predictions:
        raise FileNotFoundError(f"No predictions found at {predictions_path}. Run inference first.")
    if output_path.exists() and not overwrite and embeddings_match_predictions(output_path, predictions):
        logger.info("Embeddings already exist at %s; use --overwrite to recompute.", output_path)
        return
    if output_path.exists() and not overwrite:
        logger.warning(
            "Existing embeddings at %s do not match the current %d predictions; recomputing.",
            output_path,
            len(predictions),
        )

    pred_texts = [row["merged_prediction_statement"] for row in predictions]
    true_texts = [row["merged_true_statement"] for row in predictions]
    logger.info("Encoding %d prediction/reference pairs for %s", len(predictions), dataset)
    pred_embeddings = encode_texts(
        tokenizer,
        model,
        tqdm(pred_texts, desc=f"embed_pred:{dataset}"),
        device,
        config.EMBEDDING["batch_size"],
        config.EMBEDDING["max_length"],
    )
    true_embeddings = encode_texts(
        tokenizer,
        model,
        tqdm(true_texts, desc=f"embed_true:{dataset}"),
        device,
        config.EMBEDDING["batch_size"],
        config.EMBEDDING["max_length"],
    )
    cosines = [
        cosine_similarity(pred_embeddings[idx], true_embeddings[idx])
        for idx in range(len(predictions))
    ]
    save_pickle(
        output_path,
        {
            "dataset": dataset,
            "sample_ids": [row["sample_id"] for row in predictions],
            "pred_embeddings": pred_embeddings,
            "true_embeddings": true_embeddings,
        },
    )
    records_to_csv(
        scores_path,
        [
            {
                "dataset": dataset,
                "dataset_type": predictions[idx]["dataset_type"],
                "sample_id": predictions[idx]["sample_id"],
                "global_cosine": cosines[idx],
            }
            for idx in range(len(predictions))
        ],
    )
    write_json(
        metadata_path,
        {
            "dataset": dataset,
            "embedding_model": str(config.MODEL_PATHS["embedding_model"]),
            "batch_size": config.EMBEDDING["batch_size"],
            "max_length": config.EMBEDDING["max_length"],
            "completed_samples": len(predictions),
        },
    )
    logger.info("Saved embeddings to %s", output_path)


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = get_device(args.device)
    logger = setup_logging("run_embeddings", config.RESULTS_DIR / "run_embeddings.log")
    logger.info("Loading embedding model from %s on %s", config.MODEL_PATHS["embedding_model"], device)
    tokenizer, model = load_embedding_model(config.MODEL_PATHS["embedding_model"], device, config.TORCH_DTYPE)
    for dataset in dataset_names(args.dataset):
        run_for_dataset(dataset, tokenizer, model, device, overwrite=args.overwrite)
    clear_cuda_cache()


if __name__ == "__main__":
    main()
