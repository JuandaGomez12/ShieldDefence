"""Crea los ejemplos para el ajuste fino del LLM (formato de chat en JSONL).

Cada ejemplo tiene: el mensaje del sistema, el prompt (correo + resultado de la
LSTM + señales) y la respuesta esperada armada con plantillas a partir de las
señales del correo. Se usa la etiqueta real del dataset como decisión del
clasificador, con una confianza simulada, para que el modelo aprenda a explicar
sin contradecirla.

Uso: python training/make_explanations.py --data data/processed --n 2400
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shield_core.explain import SYSTEM, build_prompt, format_answer, template_explanation  # noqa: E402
from shield_core.signals import detect  # noqa: E402


def make(df, n, rng):
    df = df.sample(n=min(n, len(df)), random_state=rng.randint(0, 10**6))
    rows = []
    for text, label in zip(df.text.astype(str), df.label.astype(int)):
        text = text[:2500]
        sig = detect(text)
        prob = rng.uniform(0.80, 0.995) if label == 1 else rng.uniform(0.01, 0.25)
        ans = format_answer(template_explanation(text, bool(label), prob, sig, rng))
        rows.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(text, bool(label), prob, sig)},
            {"role": "assistant", "content": ans},
        ]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--out", default=str(ROOT / "data" / "explanations"))
    a = ap.parse_args()
    rng = random.Random(7)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", a.n), ("val", max(a.n // 10, 50))):
        df = pd.read_csv(Path(a.data) / f"{split}.csv")
        rows = make(df, n, rng)
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, len(rows))
    print("\nEjemplo:\n", rows[0]["messages"][1]["content"][:600], "\n---\n", rows[0]["messages"][2]["content"])


if __name__ == "__main__":
    main()
