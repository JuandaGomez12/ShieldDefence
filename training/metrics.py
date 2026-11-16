"""Métricas comunes para comparar modelos."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def report(y, p, sources=None, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).ravel()
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y)),
        "umbral": float(threshold),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "matriz": {"VN": int(tn), "FP": int(fp), "FN": int(fn), "VP": int(tp)},
    }
    if sources is not None:
        src = np.asarray(sources)
        out["por_fuente"] = {
            s: round(float((pred[src == s] == y[src == s]).mean()), 4) for s in sorted(set(src))
        }
    return out


def pick_threshold(y, p, min_precision=0.95):
    """Umbral que da el mayor recall manteniendo la precisión por encima del mínimo."""
    y = np.asarray(y)
    best_t, best_key = 0.5, (-1.0, -1.0)
    for t in np.arange(0.05, 0.96, 0.01):
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        prec = tp / max(pred.sum(), 1)
        rec = tp / max((y == 1).sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if prec >= min_precision and (rec, f1) > best_key:
            best_t, best_key = float(t), (rec, f1)
    return round(best_t, 2)
