"""Evalúa el LLM explicador ya ajustado.

Mide, sobre correos que el LLM no vio al entrenar:
  - formato_valido: % de respuestas con Explicación / Señales / Recomendación
  - coherente:      % que no contradicen la decisión de la LSTM
  - segundos:       tiempo promedio por explicación
y guarda las explicaciones de los correos en español para revisarlas a mano.

Uso: python training/eval_llm.py --model models/explicador/fusionado
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.pipeline import consistent  # noqa: E402
from shield_core.explain import SYSTEM, build_prompt, parse_answer  # noqa: E402
from shield_core.lstm_numpy import NumpyBiLSTM  # noqa: E402
from shield_core.preprocess import clean_raw, for_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "explicador" / "fusionado"))
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "models" / "eval_llm.json"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    lstm = NumpyBiLSTM(ROOT / "models")
    thr = json.load(open(ROOT / "models" / "lstm_config.json"))["umbral"]

    es = pd.read_csv(ROOT / "data" / "spanish_eval.csv").assign(grupo="español")
    data_dir = ROOT / "data" / ("processed_bilingue" if (ROOT / "data" / "processed_bilingue").exists() else "processed")
    te = pd.read_csv(data_dir / "test.csv").sample(n=a.n_test, random_state=5)
    te = te.rename(columns={"source": "note"})[["text", "label", "note"]].assign(grupo="test")
    rows = []
    for text, label, note, grupo in pd.concat([es, te]).itertuples(index=False):
        email = clean_raw(str(text))
        p = lstm.predict(for_model(email))
        is_ph = p >= thr
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(email, is_ph, p)}]
        enc = tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False),
                  return_tensors="pt", add_special_tokens=False).to(dev)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=220, do_sample=False, repetition_penalty=1.1,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        secs = time.time() - t0
        raw = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = parse_answer(raw)
        rows.append({"grupo": grupo, "nota": note, "etiqueta_real": int(label), "prob_lstm": round(p, 3),
                     "lstm_dice_phishing": bool(is_ph), "formato_valido": parsed is not None,
                     "coherente": bool(parsed and consistent(parsed, is_ph)), "segundos": round(secs, 2),
                     "tokens": int(out.shape[1] - enc["input_ids"].shape[1]), "respuesta": raw})
    df = pd.DataFrame(rows)
    resumen = {
        "n": len(df),
        "formato_valido": round(df.formato_valido.mean(), 3),
        "coherente_con_lstm": round(df.coherente.mean(), 3),
        "segundos_promedio": round(df.segundos.mean(), 2),
        "tokens_promedio": round(df.tokens.mean(), 1),
        "dispositivo": dev,
    }
    json.dump({"resumen": resumen, "detalle": rows}, open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    for r in rows[:4]:
        print("\n====", r["nota"], "| LSTM:", "phishing" if r["lstm_dice_phishing"] else "legítimo")
        print(r["respuesta"])


if __name__ == "__main__":
    main()
