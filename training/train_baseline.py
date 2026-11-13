"""Modelo base: regresión logística sobre TF-IDF (como en EXPLICATE)."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shield_core.preprocess import for_model  # noqa: E402
from training.metrics import report  # noqa: E402

import argparse  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "processed"))
D = Path(ap.parse_args().data)
train, test = pd.read_csv(D / "train.csv"), pd.read_csv(D / "test.csv")
es = pd.read_csv(ROOT / "data" / "spanish_eval.csv")

model = make_pipeline(
    TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000, C=4.0),
)
model.fit(train.model_text.fillna(""), train.label)

res = {
    "test": report(test.label, model.predict_proba(test.model_text.fillna(""))[:, 1], test.source),
    "spanish": report(es.label, model.predict_proba(es.text.map(for_model))[:, 1]),
}
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(model, ROOT / "models" / "baseline_tfidf_lr.joblib")
json.dump(res, open(ROOT / "models" / "baseline_metrics.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps({k: {m: v for m, v in r.items() if m != "por_fuente"} for k, r in res.items()}, indent=2))
print("por fuente:", json.dumps(res["test"]["por_fuente"], indent=1))
