"""Entrena el combinador (BiLSTM + señales) y compara contra la BiLSTM sola.

1. Calcula la probabilidad de la BiLSTM (versión NumPy, la misma que usa la página) para validación.
2. Ajusta una regresión logística con esa probabilidad y las señales de las reglas.
3. Elige el umbral en validación (precisión mínima, igual que la LSTM).
4. Evalúa la BiLSTM sola y el combinador en prueba y en la prueba en español escrita a mano, por categoría,
   y lista los correos que fallan.

Uso: python training/train_combiner.py --data data/processed_bilingue --extra data/synthetic_es
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shield_core.combine import FEATURES, features  # noqa: E402
from shield_core.lstm_numpy import NumpyBiLSTM  # noqa: E402
from shield_core.preprocess import clean_raw, for_model  # noqa: E402
from shield_core.signals import detect  # noqa: E402
from training.metrics import report  # noqa: E402
from training.metrics import pick_threshold  # noqa: E402


def lstm_probs(model, texts):
    return np.array([model.predict(for_model(t)) for t in texts])


def feats(texts, probs):
    return np.array([features(clean_raw(t), p, detect(clean_raw(t))) for t, p in zip(texts, probs)])


def evaluate(name, df, p, thr, show_errors=False):
    pred = p >= thr
    ok = pred == df.label.values.astype(bool)
    out = {"accuracy": round(float(ok.mean()), 4), "falsos_positivos": int((pred & (df.label.values == 0)).sum()),
           "phishing_que_se_escapa": int((~pred & (df.label.values == 1)).sum())}
    if "categoria" in df:
        out["por_categoria"] = {c: f"{int(ok[m].sum())}/{int(m.sum())}" for c in sorted(df.categoria.unique())
                                for m in [df.categoria.values == c]}
    if show_errors:
        out["errores"] = [f"[{'phishing' if l else 'legítimo'} -> {pp:.2f}] {n}"
                          for n, l, pp, o in zip(df.note, df.label, p, ok) if not o]
    print(f"\n== {name}: {json.dumps({k: v for k, v in out.items() if k != 'errores'}, ensure_ascii=False)}")
    for e in out.get("errores", []):
        print("   ", e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "processed_bilingue"))
    ap.add_argument("--extra", default="", help="carpeta con sintéticos (se usa su val.csv)")
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--min_precision", type=float, default=0.95)
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()
    M = Path(a.models)
    model = NumpyBiLSTM(M)
    lstm_thr = json.load(open(M / "lstm_config.json"))["umbral"]

    D = Path(a.data)
    va, te = pd.read_csv(D / "val.csv"), pd.read_csv(D / "test.csv")
    if a.extra:
        va = pd.concat([va, pd.read_csv(Path(a.extra) / "val.csv")], ignore_index=True)
    es = pd.read_csv(ROOT / "data" / "spanish_eval.csv")
    for d in (va, te, es):
        d["text"] = d["text"].fillna("").astype(str)

    print("Calculando la BiLSTM en validación, prueba y español...", flush=True)
    pv, pt, pe = lstm_probs(model, va.text), lstm_probs(model, te.text), lstm_probs(model, es.text)
    Xv, Xt, Xe = feats(va.text, pv), feats(te.text, pt), feats(es.text, pe)

    lr = LogisticRegression(C=a.C, max_iter=2000, class_weight="balanced")
    lr.fit(Xv, va.label.values)
    cv = lr.predict_proba(Xv)[:, 1]
    thr = pick_threshold(va.label.values, cv, a.min_precision)
    ct, ce = lr.predict_proba(Xt)[:, 1], lr.predict_proba(Xe)[:, 1]

    pesos = dict(zip(FEATURES, [round(float(w), 4) for w in lr.coef_[0]]))
    print("Pesos del combinador:", json.dumps(pesos, ensure_ascii=False))
    print("Sesgo:", round(float(lr.intercept_[0]), 4), "| umbral:", thr)

    res = {"lstm_sola": {"umbral": lstm_thr, "test": report(te.label, pt, te.source, lstm_thr),
                         "espanol": evaluate("BiLSTM sola, español", es, pe, lstm_thr, True)},
           "combinador": {"umbral": thr, "test": report(te.label, ct, te.source, thr),
                          "espanol": evaluate("Combinador, español", es, ce, thr, True)}}
    print("\nPrueba general  BiLSTM sola:", res["lstm_sola"]["test"]["accuracy"],
          "| combinador:", res["combinador"]["test"]["accuracy"])
    json.dump({"features": FEATURES, "pesos": [float(w) for w in lr.coef_[0]], "sesgo": float(lr.intercept_[0]),
               "umbral": thr}, open(M / "combinador.json", "w"), indent=2)
    json.dump(res, open(M / "combinador_eval.json", "w"), indent=2, ensure_ascii=False)
    print("Guardado en", M / "combinador.json")


if __name__ == "__main__":
    main()
