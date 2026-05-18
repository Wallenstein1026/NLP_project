from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_REFINE_DIR = OUTPUT_DIR / "results_refine"
REPORT_READY_DIR = RESULTS_DIR / "report_ready"
MODEL_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"


MODEL_PATHS = {
    "answer_model": MODEL_DIR / "Llama-3.2-3B-Instruct",
    "statement_model": MODEL_DIR / "Qwen2.5-7B-Instruct",
    "hhem_model": MODEL_DIR / "HHEM-2.1-Open",
    "hhem_foundation_model": MODEL_DIR / "flan-t5-base",
    "embedding_model": MODEL_DIR / "all-mpnet-base-v2",
    "embedding_model_alternatives": {
        "all-mpnet-base-v2": MODEL_DIR / "all-mpnet-base-v2",
        "all-MiniLM-L6-v2": MODEL_DIR / "all-MiniLM-L6-v2",  # Alternative smaller model
        "paraphrase-MiniLM-L6-v2": MODEL_DIR / "paraphrase-MiniLM-L6-v2",  # For paraphrase similarity
    },
}


DATASETS = {
    "sciq": {
        "path": DATA_DIR / "sciq" / "merged_fb.json",
        "type": "short",
        "description": "SciQ short-form science QA",
    },
    "simple_questions_wiki": {
        "path": DATA_DIR / "simple_questions_wiki" / "merged_fb.json",
        "type": "short",
        "description": "Simple Questions Wiki-based short-form QA",
    },
    "nq": {
        "path": DATA_DIR / "nq" / "merged_fb.json",
        "type": "long",
        "description": "Natural Questions long-form QA",
    },
    "truthfulQA": {
        "path": DATA_DIR / "truthfulQA" / "merged_fb.json",
        "type": "long",
        "description": "TruthfulQA long-form QA",
    },
}

DATASET_GROUPS = {
    "short": ["sciq", "simple_questions_wiki"],
    "long": ["nq", "truthfulQA"],
}


DEFAULT_PROFILE = "main"

PROFILE_SAMPLE_LIMITS = {
    "pilot": {
        "sciq": 20,
        "simple_questions_wiki": 20,
        "nq": 20,
        "truthfulQA": 20,
    },
    "main": {
        "sciq": 1000,
        "simple_questions_wiki": 1000,
        "nq": 1000,
        "truthfulQA": 1000,
    },
    "full": {
        "sciq": None,
        "simple_questions_wiki": None,
        "nq": None,
        "truthfulQA": None,
    },
}


SEED = 42
SAMPLE_STRATEGY = "seeded_shuffle"
DEVICE = "cuda:0"
TORCH_DTYPE = "float16"


GENERATION = {
    "answer_max_new_tokens": {
        "short": 24,
        "long": 128,
    },
    "statement_max_new_tokens": 192,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_input_tokens": 2048,
}

PROMPTS = {
    "answer_short": (
        "Answer the question with a concise factual answer. "
        "Use a short noun phrase or named entity when possible. Do not explain.\n"
        "Question: {question}\n"
        "Answer:"
    ),
    "answer_long": (
        "Answer the question in one clear factual sentence. "
        "Do not include unsupported details.\n"
        "Question: {question}\n"
        "Answer:"
    ),
    "statement": (
        "Convert the Q&A pair into one standalone factual statement. "
        "Keep the meaning of the answer unchanged and do not add new facts.\n"
        "Question: {question}\n"
        "Answer: {answer}\n"
        "Statement:"
    ),
}


HHEM = {
    "batch_size": 8,
    "threshold": 0.5,
    "sanity_pairs": [
        {
            "premise": "The capital of France is Paris.",
            "hypothesis": "The capital of France is Paris.",
            "label": 1,
        },
        {
            "premise": "The capital of France is Paris.",
            "hypothesis": "The capital of France is Berlin.",
            "label": 0,
        },
        {
            "premise": "I am in California.",
            "hypothesis": "I am in the United States.",
            "label": 1,
        },
        {
            "premise": "I am in the United States.",
            "hypothesis": "I am in California.",
            "label": 0,
        },
    ],
}


EMBEDDING = {
    "batch_size": 64,
    "max_length": 384,
}


ANALYSIS = {
    "threshold_min": 0.0,
    "threshold_max": 1.0,
    "threshold_step": 0.01,
    "figure_dpi": 220,
    "traditional_metrics": ["exact_match", "token_f1", "bleu"],
}


ABLATION = {
    "component_methods": [
        "global_cosine_only",
        "token_f1_only",
        "exact_match_only",
        "alignment_only",
        "semantic_components_only",
        "lexical_components_only",
        "all_components_uniform",
        "short_hybrid_full",
        "hybrid_without_exact_match",
        "hybrid_without_token_f1",
        "hybrid_without_cosine",
        "type_strategy_full",
    ],
}


IMPROVEMENT = {
    "short_hybrid_weights": {
        "exact_match": 0.35,
        "token_f1": 0.30,
        "list_set_match": 0.15,
        "cosine": 0.20,
    },
    "alignment_text_source": "answers",
    "alignment_batch_size": 64,
    "alignment_max_length": 192,
    "alignment_chunk_size": 256,
}


FAILURE_ANALYSIS = {
    "top_k_per_type": 50,
    "high_similarity_quantile": 0.80,
    "low_similarity_quantile": 0.20,
    "lexical_token_f1_max": 0.50,
    "semantic_ambiguity_token_f1_min": 0.20,
    "long_reference_word_min": 75,
}


PIPELINE_STAGES = [
    "inference",
    "correctness",
    "embeddings",
    "similarity",
    "improvement",
    "refine",
    "refinement_ablation",
    "failures",
    "ablation",
    "report_ready",
]
