"""Revisa que la prueba en español no se parezca demasiado a los correos sintéticos de entrenamiento.

Si un correo de prueba es casi igual a uno de entrenamiento, acertarlo no dice nada sobre si el
modelo generaliza. Para cada correo de prueba se busca el sintético más parecido:
  - similitud coseno con TF-IDF de palabras (1 y 2 palabras seguidas)
  - fracción de sus secuencias de 4 palabras que aparecen tal cual en algún sintético

Uso: python training/check_overlap.py --synthetic data/synthetic_es/train.csv [--max_cos 0.6]
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]


def norm(t):
    t = unicodedata.normalize("NFD", str(t).lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9ñ ]+", " ", t)


def grams(t, n=4):
    w = norm(t).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default=str(ROOT / "data" / "synthetic_es" / "train.csv"))
    ap.add_argument("--eval", default=str(ROOT / "data" / "spanish_eval.csv"))
    ap.add_argument("--max_cos", type=float, default=0.6)
    ap.add_argument("--max_4gram", type=float, default=0.3)
    a = ap.parse_args()
    syn = pd.read_csv(a.synthetic).text.astype(str)
    ev = pd.read_csv(a.eval)
    vec = TfidfVectorizer(preprocessor=norm, ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(pd.concat([syn, ev.text]))
    S, E = vec.transform(syn), vec.transform(ev.text)
    sims = (E @ S.T).toarray()
    syn_grams = set().union(*(grams(t) for t in syn))
    bad = 0
    for i, r in ev.iterrows():
        j = sims[i].argmax()
        g = grams(r.text)
        frac = len(g & syn_grams) / max(len(g), 1)
        flag = sims[i, j] >= a.max_cos or frac >= a.max_4gram
        bad += flag
        if flag:
            print(f"[parecido] coseno {sims[i, j]:.2f} | 4-gramas {frac:.0%} | {r.note}\n    sintético: {syn.iloc[j][:160]!r}")
    print(f"\n{bad} de {len(ev)} correos de prueba se parecen demasiado a los sintéticos "
          f"(coseno >= {a.max_cos} o >= {a.max_4gram:.0%} de 4-gramas iguales)")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
