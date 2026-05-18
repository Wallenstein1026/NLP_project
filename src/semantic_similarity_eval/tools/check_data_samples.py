import argparse
import random
from typing import Optional

from semantic_similarity_eval import config
from semantic_similarity_eval.utils.io import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Print random data or prediction samples for manual inspection.")
    parser.add_argument("--dataset", choices=list(config.DATASETS), default=None)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--predictions", action="store_true")
    return parser.parse_args()


def dataset_names(dataset: Optional[str]):
    return [dataset] if dataset else list(config.DATASETS.keys())


def sample_rows(rows, n):
    rng = random.Random(config.SEED)
    if len(rows) <= n:
        return rows
    return rng.sample(rows, n)


def main():
    args = parse_args()
    for dataset in dataset_names(args.dataset):
        if args.predictions:
            path = config.RESULTS_DIR / dataset / "predictions.jsonl"
        else:
            path = config.DATASETS[dataset]["path"]
        rows = read_jsonl(path)
        print(f"\n=== {dataset} | {path} | rows={len(rows)} ===")
        for row in sample_rows(rows, args.n):
            print(f"question: {row.get('question', '')}")
            if args.predictions:
                print(f"prediction: {row.get('raw_prediction', '')}")
                print(f"reference: {row.get('true_answer', '')}")
                print(f"prediction statement: {row.get('merged_prediction_statement', '')}")
                print(f"reference statement: {row.get('merged_true_statement', '')}")
            else:
                print(f"correct_answer: {row.get('correct_answer', '')}")
            print("---")


if __name__ == "__main__":
    main()

