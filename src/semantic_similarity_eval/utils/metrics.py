from typing import Dict, Iterable, List, Optional

import numpy as np


def cosine_similarity(vec_a, vec_b) -> float:
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_matrix(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return np.matmul(a_norm, b_norm.T)


def normalize_cosine_for_mixture(score: float) -> float:
    return float(np.clip(score, 0.0, 1.0))


def binary_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = max(len(y_true), 1)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def make_thresholds(start: float = 0.0, stop: float = 1.0, step: float = 0.01) -> np.ndarray:
    count = int(round((stop - start) / step)) + 1
    return np.round(np.linspace(start, stop, count), 6)


def threshold_sweep(y_true, scores, thresholds: Optional[Iterable[float]] = None) -> List[Dict[str, float]]:
    if thresholds is None:
        thresholds = make_thresholds()
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float32)
    rows = []
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        row = {"threshold": float(threshold)}
        row.update(binary_metrics(y_true, y_pred))
        rows.append(row)
    return rows


def best_threshold(y_true, scores, thresholds: Optional[Iterable[float]] = None) -> Dict[str, float]:
    rows = threshold_sweep(y_true, scores, thresholds)
    if not rows:
        return {"threshold": 0.5, "f1": 0.0}
    return max(rows, key=lambda row: (row["f1"], row["accuracy"], -row["threshold"]))


def auc_scores(y_true, scores) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float32)
    result = {"auroc": float("nan"), "auprc": float("nan")}
    if len(np.unique(y_true)) < 2:
        return result
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        result["auroc"] = float(roc_auc_score(y_true, scores))
        result["auprc"] = float(average_precision_score(y_true, scores))
    except Exception:
        return result
    return result


def curve_points(y_true, scores) -> Dict[str, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float32)
    if len(np.unique(y_true)) < 2:
        return {}
    try:
        from sklearn.metrics import precision_recall_curve, roc_curve

        fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
        return {
            "fpr": fpr,
            "tpr": tpr,
            "roc_thresholds": roc_thresholds,
            "precision": precision,
            "recall": recall,
            "pr_thresholds": pr_thresholds,
        }
    except Exception:
        return {}


def safe_corr(x, y) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": float("nan"), "spearman": float("nan")}
    pearson = float(np.corrcoef(x, y)[0, 1])
    try:
        from scipy.stats import spearmanr

        spearman = float(spearmanr(x, y).correlation)
    except Exception:
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def metric_summary(y_true, scores, thresholds=None) -> Dict[str, float]:
    best = best_threshold(y_true, scores, thresholds)
    auc = auc_scores(y_true, scores)
    row = dict(best)
    row.update(auc)
    return row

