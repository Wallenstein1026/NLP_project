# Data and Model Access README

This repository does not include the raw datasets or downloaded model checkpoints because they are too large for submission. The project can be reproduced by downloading the public datasets and pretrained models listed below.

## Datasets

| Dataset used in project | Role in project | Access link |
| --- | --- | --- |
| SciQ | Short-form science QA evaluation dataset. | https://huggingface.co/datasets/allenai/sciq |
| Simple Questions / SimpleQuestionsV2 | Short-form factoid QA evaluation dataset. | https://huggingface.co/datasets/fbougares/simple_questions_v2 |
| Natural Questions (NQ) | Long-form QA evaluation dataset. | https://github.com/google-research-datasets/natural-questions |
| TruthfulQA | Long-form truthfulness/factuality QA evaluation dataset. | https://github.com/sylinrl/TruthfulQA |

In our main experiments, we used a seeded subset profile: up to 1000 examples for SciQ, Simple Questions, and Natural Questions, and 817 processed examples for TruthfulQA.

## Models

| Model used in project | Role in project | Access link |
| --- | --- | --- |
| Meta Llama-3.2-3B-Instruct | Answer generation model. | https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct |
| Qwen2.5-7B-Instruct | Statement conversion model for rewriting QA pairs into standalone factual statements. | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct |
| Vectara HHEM-2.1-Open (`vectara/hallucination_evaluation_model`) | Entailment/factual consistency model used for automatic correctness scoring. | https://huggingface.co/vectara/hallucination_evaluation_model |
| Sentence-Transformers all-mpnet-base-v2 | Sentence embedding model used to compute semantic cosine similarity. | https://huggingface.co/sentence-transformers/all-mpnet-base-v2 |
| Google FLAN-T5-base | Foundation model dependency used by the local HHEM setup. | https://huggingface.co/google/flan-t5-base |

No model checkpoints are included in the submitted package. Please download them from the links above and place them under the local `models/` directory if running the pipeline locally.
