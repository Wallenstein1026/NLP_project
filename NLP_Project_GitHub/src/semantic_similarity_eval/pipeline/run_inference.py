import argparse
import random
from typing import Dict, List, Optional

from tqdm import tqdm

from semantic_similarity_eval import config
from semantic_similarity_eval.utils.io import append_jsonl, completed_sample_ids, dataset_result_dir, records_to_csv, read_jsonl, write_json
from semantic_similarity_eval.utils.logging_utils import setup_logging
from semantic_similarity_eval.utils.modeling import clear_cuda_cache, generate_text, get_device, load_causal_lm, set_seed
from semantic_similarity_eval.utils.text_normalize import fallback_statement, normalize_answer, word_count


def parse_args():
    parser = argparse.ArgumentParser(description="Generate QA predictions and factual statements.")
    parser.add_argument("--dataset", choices=list(config.DATASETS), default=None)
    parser.add_argument("--profile", choices=list(config.PROFILE_SAMPLE_LIMITS), default=config.DEFAULT_PROFILE)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dataset_names(dataset: Optional[str]) -> List[str]:
    if dataset:
        return [dataset]
    return list(config.DATASETS.keys())


def load_dataset_records(dataset: str, profile: str) -> List[Dict]:
    path = config.DATASETS[dataset]["path"]
    rows = read_jsonl(path)
    records = []
    for idx, row in enumerate(rows):
        records.append(
            {
                "dataset": dataset,
                "dataset_type": config.DATASETS[dataset]["type"],
                "sample_id": str(idx),
                "question": row.get("question", ""),
                "true_answer": row.get("correct_answer", row.get("answer", "")),
            }
        )
    limit = config.PROFILE_SAMPLE_LIMITS[profile][dataset]
    if limit is None or limit >= len(records):
        return records
    rng = random.Random(config.SEED)
    indices = list(range(len(records)))
    if config.SAMPLE_STRATEGY == "seeded_shuffle":
        rng.shuffle(indices)
    selected = sorted(indices[:limit])
    return [records[i] for i in selected]


def answer_prompt(dataset_type: str, question: str) -> str:
    key = "answer_short" if dataset_type == "short" else "answer_long"
    return config.PROMPTS[key].format(question=question)


def statement_prompt(question: str, answer: str) -> str:
    return config.PROMPTS["statement"].format(question=question, answer=answer)


def generate_statement(
    tokenizer,
    model,
    question: str,
    answer: str,
    device: str,
    logger,
) -> str:
    if not str(answer or "").strip():
        return fallback_statement(question, answer)
    try:
        statement = generate_text(
            tokenizer=tokenizer,
            model=model,
            prompt=statement_prompt(question, answer),
            device=device,
            max_new_tokens=config.GENERATION["statement_max_new_tokens"],
            max_input_tokens=config.GENERATION["max_input_tokens"],
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
        return statement or fallback_statement(question, answer)
    except Exception as exc:
        logger.warning("Statement generation failed; using fallback. error=%s", exc)
        return fallback_statement(question, answer)


def run_for_dataset(
    dataset: str,
    profile: str,
    device: str,
    answer_tokenizer,
    answer_model,
    statement_tokenizer,
    statement_model,
    overwrite: bool = False,
) -> None:
    result_dir = dataset_result_dir(config.RESULTS_DIR, dataset)
    logger = setup_logging("run_inference", result_dir / "inference.log")
    output_path = result_dir / "predictions.jsonl"
    csv_path = result_dir / "predictions.csv"
    metadata_path = result_dir / "prediction_metadata.json"

    records = load_dataset_records(dataset, profile)
    if overwrite and output_path.exists():
        output_path.unlink()
    done_ids = set() if overwrite else completed_sample_ids(output_path)

    logger.info("Dataset=%s profile=%s total_selected=%d already_done=%d", dataset, profile, len(records), len(done_ids))

    for record in tqdm(records, desc=f"inference:{dataset}"):
        sample_id = str(record["sample_id"])
        if sample_id in done_ids:
            continue
        dataset_type = record["dataset_type"]
        try:
            raw_prediction = generate_text(
                tokenizer=answer_tokenizer,
                model=answer_model,
                prompt=answer_prompt(dataset_type, record["question"]),
                device=device,
                max_new_tokens=config.GENERATION["answer_max_new_tokens"][dataset_type],
                max_input_tokens=config.GENERATION["max_input_tokens"],
                do_sample=config.GENERATION["do_sample"],
                temperature=config.GENERATION["temperature"],
                top_p=config.GENERATION["top_p"],
            )
        except Exception as exc:
            logger.exception("Answer generation failed for %s/%s", dataset, sample_id)
            raw_prediction = ""
            record["generation_error"] = str(exc)

        normalized_prediction = normalize_answer(raw_prediction)
        merged_prediction = generate_statement(
            statement_tokenizer,
            statement_model,
            record["question"],
            raw_prediction,
            device,
            logger,
        )
        merged_true = generate_statement(
            statement_tokenizer,
            statement_model,
            record["question"],
            record["true_answer"],
            device,
            logger,
        )

        output = {
            **record,
            "raw_prediction": raw_prediction,
            "normalized_prediction": normalized_prediction,
            "merged_prediction_statement": merged_prediction,
            "merged_true_statement": merged_true,
            "prediction_length": word_count(raw_prediction),
            "profile": profile,
        }
        append_jsonl(output_path, output)

    final_records = read_jsonl(output_path)
    records_to_csv(csv_path, final_records)
    write_json(
        metadata_path,
        {
            "dataset": dataset,
            "profile": profile,
            "selected_samples": len(records),
            "completed_samples": len(final_records),
            "sample_strategy": config.SAMPLE_STRATEGY,
            "seed": config.SEED,
            "answer_model": str(config.MODEL_PATHS["answer_model"]),
            "statement_model": str(config.MODEL_PATHS["statement_model"]),
            "generation": config.GENERATION,
        },
    )
    logger.info("Saved %d prediction records to %s", len(final_records), output_path)


def needs_inference(dataset: str, profile: str, overwrite: bool = False) -> bool:
    if overwrite:
        return True
    result_dir = dataset_result_dir(config.RESULTS_DIR, dataset)
    output_path = result_dir / "predictions.jsonl"
    records = load_dataset_records(dataset, profile)
    done_ids = completed_sample_ids(output_path)
    return any(str(record["sample_id"]) not in done_ids for record in records)


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = get_device(args.device)
    logger = setup_logging("run_inference", config.RESULTS_DIR / "run_inference.log")
    pending_datasets = [
        dataset for dataset in dataset_names(args.dataset)
        if needs_inference(dataset, args.profile, args.overwrite)
    ]
    if not pending_datasets:
        logger.info("All selected prediction files are complete for profile=%s; skipping model loading.", args.profile)
        return
    logger.info("Loading answer model from %s on %s", config.MODEL_PATHS["answer_model"], device)
    answer_tokenizer, answer_model = load_causal_lm(config.MODEL_PATHS["answer_model"], device, config.TORCH_DTYPE)
    logger.info("Loading statement model from %s on %s", config.MODEL_PATHS["statement_model"], device)
    statement_tokenizer, statement_model = load_causal_lm(config.MODEL_PATHS["statement_model"], device, config.TORCH_DTYPE)

    for dataset in pending_datasets:
        run_for_dataset(
            dataset=dataset,
            profile=args.profile,
            device=device,
            answer_tokenizer=answer_tokenizer,
            answer_model=answer_model,
            statement_tokenizer=statement_tokenizer,
            statement_model=statement_model,
            overwrite=args.overwrite,
        )
    clear_cuda_cache()


if __name__ == "__main__":
    main()
