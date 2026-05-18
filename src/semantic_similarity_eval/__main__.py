import argparse
import sys
from typing import Callable, Dict, List


COMMANDS: Dict[str, str] = {
    "pipeline": "semantic_similarity_eval.pipeline.run_pipeline:main",
    "inference": "semantic_similarity_eval.pipeline.run_inference:main",
    "correctness": "semantic_similarity_eval.pipeline.run_eval_correctness:main",
    "embeddings": "semantic_similarity_eval.pipeline.run_embeddings:main",
    "similarity": "semantic_similarity_eval.analysis.analyze_similarity:main",
    "improve": "semantic_similarity_eval.analysis.improve_metric:main",
    "refine": "semantic_similarity_eval.analysis.refine_evaluation:main",
    "refinement-ablation": "semantic_similarity_eval.analysis.refinement_ablation:main",
    "failures": "semantic_similarity_eval.analysis.analyze_failures:main",
    "embedding-ablation": "semantic_similarity_eval.analysis.embedding_ablation:main",
    "plot": "semantic_similarity_eval.analysis.plot_refined_results:main",
    "samples": "semantic_similarity_eval.tools.check_data_samples:main",
    "smoke-test": "semantic_similarity_eval.tools.smoke_tests:main",
}


ALIASES = {
    "run": "pipeline",
    "eval": "correctness",
    "analyze": "similarity",
    "ablation": "embedding-ablation",
}


def _load_callable(target: str) -> Callable[[], None]:
    module_name, function_name = target.split(":", 1)
    module = __import__(module_name, fromlist=[function_name])
    return getattr(module, function_name)


def main(argv: List[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m semantic_similarity_eval",
        description="Unified CLI for the semantic similarity evaluation project.",
    )
    parser.add_argument("command", nargs="?", help=f"One of: {', '.join(sorted(COMMANDS))}")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the selected command.")
    parsed = parser.parse_args(argv)

    if not parsed.command:
        parser.print_help()
        return

    command = ALIASES.get(parsed.command, parsed.command)
    if command not in COMMANDS:
        parser.error(f"unknown command '{parsed.command}'. Valid commands: {', '.join(sorted(COMMANDS))}")

    sys.argv = [f"semantic_similarity_eval {command}", *parsed.args]
    _load_callable(COMMANDS[command])()


if __name__ == "__main__":
    main()

