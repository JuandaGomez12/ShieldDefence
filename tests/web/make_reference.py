"""Genera las salidas de Python que tests/web/parity.mjs compara con JavaScript."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shield_core.preprocess import for_model
from shield_core.signals import detect
from shield_core.explain import build_prompt, parse_answer, recommendation, needs_review, _REC_LEGIT, _REC_PHISH, _REC_MONEY
from shield_core.lstm_numpy import NumpyBiLSTM
from shield_core.combine import Combiner
from shield_core.ocrfix import fix_ocr_links

df = pd.read_csv(ROOT / "data/processed/test.csv").sample(300, random_state=1)
es = pd.read_csv(ROOT / "data/spanish_eval.csv")
texts = list(df["text"].astype(str)) + list(es["text"].astype(str))
# Versiones disfrazadas ("c l a v e", "cl4ve", invisibles, cirílicas) para probar deobfuscate en los dos lados.
import numpy as np  # noqa: E402
from training.train_combiner import obfuscate  # noqa: E402
rng = np.random.default_rng(0)
texts += [obfuscate(t, rng) for t in es["text"].astype(str)] + [
    "Confirme su ｃｌａｖｅ y su 𝐜𝐨𝐧𝐭𝐫𝐚𝐬𝐞ñ𝐚 en http://раypal.com/login", "S.A.S. E.P.S. mp3 4G covid19 B-201 FE-20391",
    # textos extremos que antes hacían lentísima la limpieza
    "hola " + "_" * 3000 + " texto unsubscribe", "hola " + "-" * 3000 + " original message --- fin", "a@" * 2000,
    "[lista] Asunto\n---- Original Message ----\nFrom: alguien\nTexto reenviado"]
MODELS = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "models"
sys.path.insert(0, "/tmp/claude-0/s")
try:
    from ia3 import IA3
    from user15 import EMAILS
    texts += IA3 + EMAILS
except ImportError:
    pass
texts += pd.read_csv(ROOT / "data/spanish_blind3.csv").text.astype(str).tolist()
texts += pd.read_csv(ROOT / "data/spanish_blind.csv").text.astype(str).tolist() + pd.read_csv(ROOT / "data/spanish_blind2.csv").text.astype(str).tolist()
m = NumpyBiLSTM(MODELS)
comb = Combiner.load(MODELS)
out = []
for t in texts:
    sig = detect(t)
    p = float(m.predict(for_model(t)))
    out.append({"text": t, "model": for_model(t), "keys": [s.key for s in sig],
                "evidence": [s.evidence for s in sig], "prob": p,
                "prompt": build_prompt(t, p >= 0.5, p, sig),
                "comb": comb.prob(t, p, sig) if comb else None,
                "review": needs_review(sig, False),
                "rec": [r if r.split(" Además")[0] not in _REC_LEGIT + _REC_PHISH + _REC_MONEY else None
                        for r in (recommendation(t, sig, True), recommendation(t, sig, False))]})
OCR_CASES = ["Soporte <alertas@bancolombia-soporte top>", "https:/Ibancolombia-soporte top/personas/validar",
             "https: //nequi-ayuda com co/login", "visita www, davivienda com", "La reunión es en el salón B top.",
             "HTTPS:/lsitio-falso xyz/LOGIN y correo pepe@empresa com", "https //bancolombia-verificacion top/login",
             "https; //nequi-ayuda com/x", "Servidor HTTP/1.1 caído en http.//sitio xyz"]
out.append({"ocr": [[c, fix_ocr_links(c)] for c in OCR_CASES]})
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
print(len(out), "casos")
