import random
from pathlib import Path
from typing import Iterable, List


def set_seed(seed: int) -> None:
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_device(requested: str) -> str:
    try:
        import torch

        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device {requested}, but this Python environment cannot see CUDA. "
                "Install a CUDA-enabled PyTorch build or fix CUDA_VISIBLE_DEVICES before running large-model stages."
            )
    except Exception:
        if requested.startswith("cuda"):
            raise
        return "cpu"
    return requested


def _torch_dtype(dtype_name: str, device: str):
    import torch

    if device == "cpu":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return torch.float16


def load_causal_lm(model_path: Path, device: str, dtype_name: str = "float16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=_torch_dtype(dtype_name, device),
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    return tokenizer, model


def load_embedding_model(model_path: Path, device: str, dtype_name: str = "float16"):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(model_path),
        torch_dtype=_torch_dtype(dtype_name, device),
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    return tokenizer, model


def load_hhem_model(model_path: Path, foundation_model_path, device: str):
    from transformers import AutoConfig, AutoModelForSequenceClassification

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    # HHEM's remote-code wrapper loads the T5 foundation from config.foundation.
    config.foundation = str(foundation_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        config=config,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    return model


def generate_text(
    tokenizer,
    model,
    prompt: str,
    device: str,
    max_new_tokens: int,
    max_input_tokens: int,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    import torch

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = max(temperature, 1e-6)
    with torch.no_grad():
        output = model.generate(**encoded, **generation_kwargs)
    new_tokens = output[0][encoded["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def mean_pool_embeddings(model_output, attention_mask):
    import torch
    import torch.nn.functional as F

    token_embeddings = model_output[0]
    expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = torch.sum(token_embeddings * expanded, dim=1) / torch.clamp(expanded.sum(dim=1), min=1e-9)
    return F.normalize(pooled, p=2, dim=1)


def encode_texts(
    tokenizer,
    model,
    texts: Iterable[str],
    device: str,
    batch_size: int,
    max_length: int,
):
    import numpy as np
    import torch

    texts = [str(text or "") for text in texts]
    vectors: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            model_output = model(**encoded)
        pooled = mean_pool_embeddings(model_output, encoded["attention_mask"])
        vectors.append(pooled.detach().cpu().numpy())
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(vectors, axis=0).astype(np.float32)


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
