"""Decisión final: junta la probabilidad de la BiLSTM con las señales de las reglas.

La BiLSTM ve el estilo del correo; las reglas ven cosas concretas (dominios falsos, pedir la clave,
pedir tarjetas de regalo, pedir discreción). Una regresión logística pequeña aprende cuánto pesa cada
cosa. Sirve sobre todo en dos casos que la LSTM sola fallaba:
  - fraude del jefe sin enlaces: no se parece a los correos de phishing del dataset, pero las reglas
    ven que pide dinero y discreción
  - correos normales en español que la LSTM ve "raros": si no piden nada ni traen enlaces sospechosos,
    el combinador baja la probabilidad

Los pesos se guardan en models/combinador.json (los entrena training/train_combiner.py).
"""
import json
import math
from pathlib import Path

from .signals import detect, find_urls

SIGNAL_KEYS = ["mixed_domain", "brand_lookalike", "security_domain", "shortener", "odd_tld", "ip_url", "no_https",
               "urgency", "threat", "credentials", "generic_greeting", "prize", "money_request", "shared_doc",
               "secrecy", "deadline", "click"]
FEATURES = ["lstm_logit"] + SIGNAL_KEYS + ["n_urls"]


def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def features(email, p_lstm, signals=None):
    signals = detect(email) if signals is None else signals
    keys = {s.key for s in signals}
    f = {"lstm_logit": logit(p_lstm)}
    for k in SIGNAL_KEYS:
        f[k] = 1.0 if k in keys else 0.0
    f["n_urls"] = float(min(len(find_urls(email)), 3))
    return [f[k] for k in FEATURES]


class Combiner:
    def __init__(self, path):
        cfg = json.load(open(path, encoding="utf-8"))
        assert cfg["features"] == FEATURES, "el combinador se entrenó con otras variables"
        self.w = cfg["pesos"]
        self.b = cfg["sesgo"]
        self.threshold = cfg["umbral"]

    def prob(self, email, p_lstm, signals=None):
        x = features(email, p_lstm, signals)
        z = self.b + sum(w * v for w, v in zip(self.w, x))
        return 1 / (1 + math.exp(-z))

    @classmethod
    def load(cls, models_dir):
        p = Path(models_dir) / "combinador.json"
        return cls(p) if p.exists() else None
