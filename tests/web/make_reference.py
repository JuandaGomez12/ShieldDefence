"""Genera las salidas de Python que tests/web/parity.mjs compara con JavaScript."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shield_core.preprocess import for_model
from shield_core.signals import detect
from shield_core.explain import build_prompt, parse_answer
from shield_core.lstm_numpy import NumpyBiLSTM

df = pd.read_csv(ROOT / "data/processed/test.csv").sample(300, random_state=1)
es = pd.read_csv(ROOT / "data/spanish_eval.csv")
texts = list(df["text"].astype(str)) + list(es["text"].astype(str))
m = NumpyBiLSTM(ROOT / "models")
out = []
for t in texts:
    sig = detect(t)
    p = float(m.predict(for_model(t)))
    out.append({"text": t, "model": for_model(t), "keys": [s.key for s in sig],
                "evidence": [s.evidence for s in sig], "prob": p,
                "prompt": build_prompt(t, p >= 0.5, p, sig)})
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
print(len(out), "casos")
