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
