import argparse
import subprocess
import sys
from typing import List

from semantic_similarity_eval import config
from semantic_similarity_eval.utils.io import ensure_dir
from semantic_similarity_eval.utils.logging_utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full semantic similarity research pipeline.")
    parser.add_argument("--dataset", choices=list(config.DATASETS), default=None)
    parser.add_argument("--profile", choices=list(config.PROFILE_SAMPLE_LIMITS), default=config.DEFAULT_PROFILE)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--start-stage", choices=config.PIPELINE_STAGES, default="inference")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stage_commands(args) -> List[List[str]]:
    dataset_args = ["--dataset", args.dataset] if args.dataset else []
    overwrite_args = ["--overwrite"] if args.overwrite else []
    report_args = [*dataset_args, "--report-ready-only"]
    return [
        [sys.executable, str(config.SCRIPTS_DIR / "run_inference.py"), "--profile", args.profile, "--device", args.device, *dataset_args, *overwrite_args],
        [sys.executable, str(config.SCRIPTS_DIR / "run_eval_correctness.py"), "--device", args.device, *dataset_args, *overwrite_args],
        [sys.executable, str(config.SCRIPTS_DIR / "run_embeddings.py"), "--device", args.device, *dataset_args, *overwrite_args],
        [sys.executable, str(config.SCRIPTS_DIR / "analyze_similarity.py"), *dataset_args],
        [sys.executable, str(config.SCRIPTS_DIR / "improve_metric.py"), "--device", args.device, *dataset_args, *overwrite_args],
        [sys.executable, str(config.SCRIPTS_DIR / "refine_evaluation.py"), "--device", args.device, *dataset_args, *overwrite_args],
        [sys.executable, str(config.SCRIPTS_DIR / "refinement_ablation.py"), "--device", args.device, *dataset_args, *overwrite_args],
        [sys.executable, str(config.SCRIPTS_DIR / "analyze_failures.py"), *dataset_args],
        [sys.executable, str(config.SCRIPTS_DIR / "embedding_ablation.py"), "--device", args.device, *dataset_args, *overwrite_args],
        [sys.executable, str(config.SCRIPTS_DIR / "analyze_failures.py"), *report_args],
    ]


def main():
    args = parse_args()
    ensure_dir(config.RESULTS_DIR)
    logger = setup_logging("run_pipeline", config.RESULTS_DIR / "run_pipeline.log")
    commands = stage_commands(args)
    stage_to_index = {stage: idx for idx, stage in enumerate(config.PIPELINE_STAGES)}
    start = stage_to_index[args.start_stage]
    for stage, command in zip(config.PIPELINE_STAGES[start:], commands[start:]):
        logger.info("Running stage=%s command=%s", stage, " ".join(command))
        subprocess.run(command, cwd=config.PROJECT_ROOT, check=True)
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
