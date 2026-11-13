"""Flujo completo de ShieldDefence: texto -> LSTM -> LLM -> respuesta.

Sigue la arquitectura del documento: la LSTM decide (clase y y confianza c) y
el LLM solo explica. El LLM recibe el correo original T, y y c.
"""
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from pathlib import Path

from shield_core.explain import SYSTEM, build_prompt, parse_answer, template_explanation
from shield_core.lstm_numpy import NumpyBiLSTM
from shield_core.preprocess import clean_raw, for_model
from shield_core.signals import detect

log = logging.getLogger("shield")
ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.getenv("SHIELD_MODELS_DIR", ROOT / "models"))
LLM_ID = os.getenv("SHIELD_LLM_MODEL", "").strip()      # p. ej. juandaGomez/shielddefence-explicador
LLM_TIMEOUT = float(os.getenv("SHIELD_LLM_TIMEOUT", "25"))
LLM_MAX_TOKENS = int(os.getenv("SHIELD_LLM_MAX_TOKENS", "220"))


class Classifier:
    """La BiLSTM entrenada con Keras, exportada a NumPy (models/lstm_pesos.npz)."""

    def __init__(self):
        cfg = json.load(open(MODELS / "lstm_config.json", encoding="utf-8"))
        self.threshold = float(cfg.get("umbral", 0.5))
        self.model = NumpyBiLSTM(MODELS)
        self.info = {"umbral": self.threshold, "f1_test": cfg.get("test", {}).get("f1"),
                     "f1_espanol": cfg.get("spanish", {}).get("f1")}

    def predict(self, text: str) -> float:
        return self.model.predict(for_model(text))


class Explainer:
    """El LLM ajustado con LoRA. Si no está disponible, se usan plantillas."""

    def __init__(self):
        self.ready = False
        self.error = None
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        if LLM_ID:
            threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            t0 = time.time()
            self.tok = AutoTokenizer.from_pretrained(LLM_ID)
            self.model = AutoModelForCausalLM.from_pretrained(LLM_ID, dtype=torch.float32)
            self.model.eval()
            self.ready = True
            log.info("LLM %s cargado en %.1fs", LLM_ID, time.time() - t0)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            log.warning("No se pudo cargar el LLM (%s); se usarán plantillas.", self.error)

    def _generate(self, prompt: str) -> str:
        import torch
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
        with self.lock, torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=LLM_MAX_TOKENS, do_sample=False, repetition_penalty=1.1,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def explain(self, email, is_phishing, prob, signals):
        if self.ready:
            try:
                raw = self.pool.submit(self._generate, build_prompt(email, is_phishing, prob, signals)).result(LLM_TIMEOUT)
                parsed = parse_answer(raw)
                if parsed and consistent(parsed, is_phishing):
                    return parsed, "llm"
                log.info("Salida del LLM descartada (formato o contradicción): %r", raw[:200])
            except FutTimeout:
                log.info("El LLM tardó más de %.0fs; se usa plantilla.", LLM_TIMEOUT)
            except Exception as e:  # noqa: BLE001
                log.warning("Error del LLM: %r", e)
        return template_explanation(email, is_phishing, prob, signals), "plantilla"


def consistent(parsed: dict, is_phishing: bool) -> bool:
    """El LLM no puede contradecir a la LSTM (regla del diseño)."""
    text = (parsed["explicacion"] + " " + parsed["recomendacion"]).lower()
    if is_phishing:
        return not re.search(r"\b(es|parece) (un correo )?leg[ií]timo\b|no (es|parece) phishing|no hay se[ñn]ales", text)
    return not re.search(r"\b(es|parece) (un )?(intento de )?phishing\b|es (un )?fraude|es una estafa", text)


class ShieldDefence:
    def __init__(self):
        self.classifier = Classifier()
        self.explainer = Explainer()

    def status(self):
        return {
            "lstm": self.classifier.info,
            "llm": {"modelo": LLM_ID or None, "listo": self.explainer.ready, "error": self.explainer.error},
        }

    def analyze(self, text: str) -> dict:
        t0 = time.time()
        email = clean_raw(text)
        prob = self.classifier.predict(email)
        is_phishing = prob >= self.classifier.threshold
        signals = detect(email)
        t1 = time.time()
        exp, source = self.explainer.explain(email, is_phishing, prob, signals)
        conf = prob if is_phishing else 1 - prob
        if is_phishing:
            level, label = ("high", "Riesgo alto") if prob >= 0.85 else ("mid", "Riesgo medio")
        else:
            level, label = "safe", "Sin señales claras de riesgo"
        return {
            "es_phishing": bool(is_phishing),
            "probabilidad_phishing": round(prob, 4),
            "confianza": round(conf, 4),
            "nivel": level,
            "etiqueta": label,
            "senales_reglas": [{"clave": s.key, "titulo": s.title, "detalle": s.detail} for s in signals],
            "explicacion": exp,
            "fuente_explicacion": source,
            "tiempos_ms": {"lstm": round((t1 - t0) * 1000), "total": round((time.time() - t0) * 1000)},
        }
