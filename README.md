# Semantic Similarity Evaluation for LLM Answers

This repository contains the complete code for an NLP project that evaluates whether semantic similarity in embedding space can proxy factual correctness for LLM-generated answers.

Large artifacts are intentionally excluded from GitHub:

- downloaded model checkpoints
- raw or processed dataset files
- generated logs, tables, embeddings, and figures

Public dataset and model links are included below.

## Clean Repository Layout

```text
.
├── scripts/                         # command-line entrypoints
├── src/semantic_similarity_eval/     # importable project package
│   ├── config.py                     # centralized paths and experiment settings
│   ├── pipeline/                     # inference, HHEM scoring, embeddings, full runner
│   ├── analysis/                     # metrics, refinement, ablations, failure analysis, plots
│   ├── tools/                        # lightweight sample inspection and smoke tests
│   └── utils/                        # shared IO, modeling, metrics, text normalization
├── data/processed/                   # local processed JSONL datasets, not tracked
├── models/                           # local model checkpoints, not tracked
├── outputs/                          # generated experiment outputs, not tracked
└── docs/                             # report/poster sources and data/model notes
```

The main path configuration lives in `src/semantic_similarity_eval/config.py`:

- datasets: `data/processed/<dataset>/merged_fb.json`
- models: `models/<model-name>/`
- outputs: `outputs/results/` and `outputs/results_refine/`

## Environment

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

The large-model stages expect a CUDA-capable PyTorch installation. Lightweight checks can run on CPU.

## Datasets

The repository does not include dataset files. Download the public datasets and preprocess them into JSONL files with fields `question` and `correct_answer`.

Expected local paths:

```text
data/processed/sciq/merged_fb.json
data/processed/simple_questions_wiki/merged_fb.json
data/processed/nq/merged_fb.json
data/processed/truthfulQA/merged_fb.json
```

Although the filenames end in `.json`, the code reads them as one JSON object per line.

| Dataset | Role | Access link |
| --- | --- | --- |
| SciQ | Short-form science QA | https://huggingface.co/datasets/allenai/sciq |
| Simple Questions / SimpleQuestionsV2 | Short-form factoid QA | https://huggingface.co/datasets/fbougares/simple_questions_v2 |
| Natural Questions | Long-form QA | https://github.com/google-research-datasets/natural-questions |
| TruthfulQA | Long-form truthfulness/factuality QA | https://github.com/sylinrl/TruthfulQA |

In the main experiments, we used a seeded subset profile: up to 1000 examples for SciQ, Simple Questions, and Natural Questions, and 817 processed examples for TruthfulQA.

## Models

No model checkpoints are included. Download the pretrained models and place them under `models/`, or edit `MODEL_PATHS` in `src/semantic_similarity_eval/config.py`.

Expected default paths:

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

## Running

Inspect dataset samples after placing processed data:

```bash
python scripts/check_data_samples.py --dataset sciq --n 3
```

Run a small pilot:

```bash
python scripts/run_pipeline.py --profile pilot
```

Run the default main subset:

```bash
python scripts/run_pipeline.py
```

Run one dataset:

```bash
python scripts/run_pipeline.py --dataset truthfulQA --profile main
```

Run all available records:

```bash
python scripts/run_pipeline.py --profile full
```

Resume is enabled by default for JSONL outputs. Add `--overwrite` only when recomputing a stage from scratch.

## Individual Stages

```bash
python scripts/run_inference.py --profile main
python scripts/run_eval_correctness.py
python scripts/run_embeddings.py
python scripts/analyze_similarity.py
python scripts/improve_metric.py
python scripts/refine_evaluation.py
python scripts/refinement_ablation.py
python scripts/analyze_failures.py
python scripts/embedding_ablation.py
python scripts/plot_refined_results.py
```

Compatibility entrypoints are preserved in `scripts/`:

- `llama_Inference.py`
- `llama_inference.py`
- `eval_hem.py`
- `encoder_embedding.py`

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
python -m compileall scripts src
python scripts/smoke_tests.py
```

The smoke tests do not load large models. Dataset checks are skipped when processed data files are absent.

Full validation after downloading datasets and models:

```bash
python scripts/run_pipeline.py --profile pilot
```
