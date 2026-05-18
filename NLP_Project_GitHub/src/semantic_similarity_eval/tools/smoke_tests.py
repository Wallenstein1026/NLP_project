import importlib.util
import math

from semantic_similarity_eval import config
from semantic_similarity_eval.utils.io import read_jsonl
from semantic_similarity_eval.utils.text_normalize import normalize_answer, token_f1


def test_datasets_load():
    missing = []
    for dataset, meta in config.DATASETS.items():
        if not meta["path"].exists():
            missing.append(dataset)
            continue
        rows = read_jsonl(meta["path"])
        if not rows:
            missing.append(dataset)
        required = {"question", "correct_answer"}
        if rows and not required.issubset(rows[0]):
            raise AssertionError(f"{dataset} missing expected fields: {required - set(rows[0])}")
    if missing:
        print(f"Skipping dataset smoke check for missing/empty datasets: {missing}")


def test_text_metrics():
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"
    assert token_f1("Paris", "Paris") == 1.0
    assert token_f1("New York City", "New York") > 0.0


def test_metric_edges():
    if importlib.util.find_spec("numpy") is None:
        print("Skipping metric edge tests because numpy is not installed in this environment.")
        return
    from semantic_similarity_eval.utils.metrics import metric_summary, threshold_sweep

    rows = threshold_sweep([1, 1], [0.2, 0.8], [0.5])
    assert rows[0]["tp"] == 1
    one_class = metric_summary([1, 1], [0.2, 0.8], [0.5])
    assert "auroc" in one_class
    assert math.isnan(one_class["auroc"])


def main():
    test_datasets_load()
    test_text_metrics()
    test_metric_edges()
    print("Smoke tests passed.")


if __name__ == "__main__":
    main()
