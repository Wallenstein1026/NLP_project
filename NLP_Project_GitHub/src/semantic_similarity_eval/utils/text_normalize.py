import re
import string
import unicodedata
from collections import Counter
from typing import List


_PUNCT_TABLE = str.maketrans({p: " " for p in string.punctuation})
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_LEAKAGE_MARKER_RE = re.compile(
    r"(?:^|[^A-Za-z0-9_])(?:Human|Assistant|Question|Answer|Context|Standalone factual statement)\s*:",
    flags=re.IGNORECASE,
)
_LIST_SPLIT_RE = re.compile(r"\s*(?:,|;|/|\band\b|\bor\b)\s+", flags=re.IGNORECASE)


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def simple_tokenize(text: str) -> List[str]:
    normalized = normalize_answer(text)
    if not normalized:
        return []
    return normalized.split()


def normalized_exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = simple_tokenize(prediction)
    ref_tokens = simple_tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def clean_generated_statement(text: str, fallback: str = "") -> tuple[str, bool]:
    """Remove common prompt/template leakage from generated standalone statements."""
    original = str(text or "").strip()
    if not original:
        return str(fallback or "").strip(), False
    match = _LEAKAGE_MARKER_RE.search(original)
    if not match:
        return original, False
    cleaned = original[:match.start()].rstrip(" .!?;:")
    if not cleaned:
        cleaned = str(fallback or "").strip()
    return cleaned, cleaned != original


def answer_items(text: str) -> List[str]:
    if text is None:
        return []
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    if not text:
        return []
    raw_parts = [part.strip() for part in _LIST_SPLIT_RE.split(text) if part.strip()]
    parts = [normalize_answer(part) for part in raw_parts]
    parts = [part for part in parts if part]
    return parts if len(parts) > 1 else []


def list_set_match(prediction: str, reference: str) -> float:
    pred_items = set(answer_items(prediction))
    ref_items = set(answer_items(reference))
    if len(pred_items) < 2 or len(ref_items) < 2:
        return 0.0
    return float(pred_items == ref_items)


def split_sentences(text: str) -> List[str]:
    if text is None:
        return []
    text = _SPACE_RE.sub(" ", str(text).strip())
    if not text:
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if len(parts) <= 1 and len(text.split()) > 80:
        chunks = []
        words = text.split()
        for start in range(0, len(words), 40):
            chunks.append(" ".join(words[start:start + 40]))
        return chunks
    return parts


def word_count(text: str) -> int:
    if text is None:
        return 0
    return len(str(text).split())


def fallback_statement(question: str, answer: str) -> str:
    question = str(question or "").strip()
    answer = str(answer or "").strip()
    if not answer:
        return question
    return f"For the question '{question}', the answer is {answer}."
