"""Arma el dataset de ShieldDefence a partir de las fuentes públicas.

Legítimos: Enron (solo los que no son spam), SpamAssassin (easy_ham + hard_ham)
           y correos legítimos generados con LLM.
Phishing:  Nazario, Nigerian Fraud y correos de phishing generados con LLM.

Salida: data/processed/{train,val,test}.csv con columnas text, label, source.
"""
import csv
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shield_core.preprocess import clean_raw, for_model  # noqa: E402

csv.field_size_limit(10**9)
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SEED = 42


def load(name, keep_label, label, sub="subject", body="body"):
    d = pd.read_csv(RAW / name, engine="python")
    if keep_label is not None:
        d = d[d["label"] == keep_label]
    text = (d[sub].fillna("").astype(str) + "\n" + d[body].fillna("").astype(str)) if sub else d[body].astype(str)
    return pd.DataFrame({"text": text, "label": label, "source": Path(name).stem})


def main():
    parts = [
        load("Enron.csv", 0, 0),
        load("SpamAssasin.csv", 0, 0),
        load("llm_legit.csv", None, 0, sub=None, body="text").assign(source="LLM_legit"),
        load("Nazario.csv", None, 1),
        load("Nigerian_Fraud.csv", None, 1),
        load("llm_phishing.csv", None, 1, sub=None, body="text").assign(source="LLM_phishing"),
    ]
    df = pd.concat(parts, ignore_index=True)
    df["text"] = df["text"].map(clean_raw)
    df["model_text"] = df["text"].map(for_model)
    before = len(df)
    df = df[df["model_text"].str.split().str.len() >= 5]
    df = df.drop_duplicates("model_text")
    print(f"Limpieza: {before} -> {len(df)} correos (vacíos y duplicados fuera)")

    # Balanceo: tantos legítimos como phishing. Se usan todos los legítimos
    # hechos con LLM y el resto se reparte por igual entre Enron y SpamAssassin,
    # para que el modelo no aprenda el estilo de una sola fuente.
    phish = df[df.label == 1]
    llm_legit = df[df.source == "LLM_legit"]
    half = (len(phish) - len(llm_legit)) // 2
    sa = df[df.source == "SpamAssasin"].sample(n=half, random_state=SEED)
    enron = df[df.source == "Enron"].sample(n=len(phish) - len(llm_legit) - half, random_state=SEED)
    data = pd.concat([phish, llm_legit, sa, enron]).sample(frac=1, random_state=SEED)

    print("\nCorreos por fuente:")
    print(data.groupby(["label", "source"]).size().to_string())
    print("\nTotal por clase:", data.label.value_counts().to_dict())

    strat = data["source"]
    train, rest = train_test_split(data, test_size=0.30, random_state=SEED, stratify=strat)
    val, test = train_test_split(rest, test_size=0.50, random_state=SEED, stratify=rest["source"])
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[["text", "model_text", "label", "source"]].to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} ({part.label.mean():.1%} phishing)")


if __name__ == "__main__":
    main()
