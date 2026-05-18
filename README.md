# Semantic Similarity Evaluation for LLM Answers

This repository contains the complete code for an NLP project that evaluates whether embedding-space semantic similarity can proxy factual correctness for LLM-generated answers.

The public GitHub version intentionally excludes large local artifacts:

- downloaded model checkpoints
- raw or processed dataset files
- generated logs, embeddings, tables, figures, and other experiment outputs

All code lives under `src/`; report and poster sources live under `docs/`.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── docs/
│   ├── report/
│   └── poster/
└── src/
    └── semantic_similarity_eval/
        ├── __main__.py             # unified CLI
        ├── config.py               # paths, model choices, profiles, thresholds
        ├── pipeline/               # inference, HHEM scoring, embeddings, full runner
        ├── analysis/               # metrics, refinement, ablations, failure analysis, plots
        ├── tools/                  # sample inspection and smoke tests
        └── utils/                  # shared IO, metrics, modeling, text normalization
```

Runtime directories are created locally when needed and are ignored by Git:

- `data/processed/`
- `models/`
- `outputs/`

The path configuration is centralized in `src/semantic_similarity_eval/config.py`.

## Environment

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

Run commands from the repository root with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m semantic_similarity_eval --help
```

The large-model stages expect a CUDA-capable PyTorch installation. Lightweight checks can run on CPU.

## Data and Model Setup

This repository does not include full data files or model checkpoints. Download them from the public sources below.

Expected processed dataset paths:

```text
data/processed/sciq/merged_fb.json
data/processed/simple_questions_wiki/merged_fb.json
data/processed/nq/merged_fb.json
data/processed/truthfulQA/merged_fb.json
```

Each dataset file is JSONL, even though the project filename ends with `.json`. Each line should contain at least:

```json
{"question": "...", "correct_answer": "..."}
```

| Dataset | Role | Access link |
| --- | --- | --- |
| SciQ | Short-form science QA | https://huggingface.co/datasets/allenai/sciq |
| Simple Questions / SimpleQuestionsV2 | Short-form factoid QA | https://huggingface.co/datasets/fbougares/simple_questions_v2 |
| Natural Questions | Long-form QA | https://github.com/google-research-datasets/natural-questions |
| TruthfulQA | Long-form truthfulness/factuality QA | https://github.com/sylinrl/TruthfulQA |

Expected model paths:

```text
models/Llama-3.2-3B-Instruct
models/Qwen2.5-7B-Instruct
models/HHEM-2.1-Open
models/flan-t5-base
models/all-mpnet-base-v2
models/all-MiniLM-L6-v2
models/paraphrase-MiniLM-L6-v2
```

| Model | Role | Access link |
| --- | --- | --- |
| Meta Llama-3.2-3B-Instruct | Answer generation | https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct |
| Qwen2.5-7B-Instruct | QA-to-statement conversion | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct |
| Vectara HHEM-2.1-Open / hallucination evaluation model | Automatic factual consistency scoring | https://huggingface.co/vectara/hallucination_evaluation_model |
| Google FLAN-T5-base | Foundation dependency for local HHEM setup | https://huggingface.co/google/flan-t5-base |
| Sentence-Transformers all-mpnet-base-v2 | Main embedding model | https://huggingface.co/sentence-transformers/all-mpnet-base-v2 |
| Sentence-Transformers all-MiniLM-L6-v2 | Embedding ablation model | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| Sentence-Transformers paraphrase-MiniLM-L6-v2 | Embedding ablation model | https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2 |

You can change local dataset or model locations in `src/semantic_similarity_eval/config.py`.

## Unified Interface

All commands use the same interface:

```bash
PYTHONPATH=src python -m semantic_similarity_eval <command> [options]
```

Common commands:

```bash
PYTHONPATH=src python -m semantic_similarity_eval samples --dataset sciq --n 3
PYTHONPATH=src python -m semantic_similarity_eval pipeline --profile pilot
PYTHONPATH=src python -m semantic_similarity_eval pipeline
PYTHONPATH=src python -m semantic_similarity_eval pipeline --dataset truthfulQA --profile main
PYTHONPATH=src python -m semantic_similarity_eval pipeline --profile full
```

Individual stages:

```bash
PYTHONPATH=src python -m semantic_similarity_eval inference --profile main
PYTHONPATH=src python -m semantic_similarity_eval correctness
PYTHONPATH=src python -m semantic_similarity_eval embeddings
PYTHONPATH=src python -m semantic_similarity_eval similarity
PYTHONPATH=src python -m semantic_similarity_eval improve
PYTHONPATH=src python -m semantic_similarity_eval refine
PYTHONPATH=src python -m semantic_similarity_eval refinement-ablation
PYTHONPATH=src python -m semantic_similarity_eval failures
PYTHONPATH=src python -m semantic_similarity_eval embedding-ablation
PYTHONPATH=src python -m semantic_similarity_eval plot
```

Resume is enabled by default for JSONL outputs. Add `--overwrite` when recomputing a stage from scratch.

## Outputs

Generated outputs are written to `outputs/` and ignored by Git:

- `outputs/results/<dataset>/predictions.jsonl`
- `outputs/results/<dataset>/correctness.jsonl`
- `outputs/results/<dataset>/embeddings.pkl`
- `outputs/results/<dataset>/similarity_scores.csv`
- `outputs/results/<dataset>/refined_similarity_scores.csv`
- `outputs/results_refine/` tables and figures

## Methods

Baseline:

- Global cosine similarity between prediction statement embeddings and ground-truth statement embeddings.

Correctness labels:

- HHEM score with premise = ground-truth statement and hypothesis = prediction statement.
- HHEM score 1 means fully supported and 0 means unsupported.

Improved metrics:

- Short-answer hybrid: normalized exact match, token F1, list-set match, and global cosine.
- Long-answer alignment: sentence splitting, pairwise embedding cosine, precision-like and recall-like max alignment, and F1-style aggregation.

Refined evaluation:

- Cleans prompt/template leakage from generated statements.
- Applies deterministic correctness overrides for exact match, list-set match, and high token F1.
- Compresses Natural Questions references by selecting relevant chunks with the embedding model.

## Validation

Lightweight checks:

```bash
python -m compileall src
PYTHONPATH=src python -m semantic_similarity_eval smoke-test
```

The smoke tests do not load large models. Dataset checks are skipped when processed data files are absent.

Full validation after downloading datasets and models:

```bash
PYTHONPATH=src python -m semantic_similarity_eval pipeline --profile pilot
```
