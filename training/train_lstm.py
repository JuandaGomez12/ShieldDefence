"""Entrena la BiLSTM de ShieldDefence.

Uso:  python training/train_lstm.py [--epochs 8] [--data data/processed]

Guarda en models/:
  lstm.keras           el modelo completo (incluye la capa de vocabulario)
  lstm_config.json     umbral elegido con validación y métricas finales
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import keras  # noqa: E402
from keras import layers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shield_core.preprocess import for_model  # noqa: E402
from training.metrics import report  # noqa: E402

VOCAB = 30000
MAX_LEN = 300
EMB = 128
UNITS = 64

# Saludos y despedidas que se agregan al azar a los correos de entrenamiento, sin importar la clase.
# En los datos originales casi solo el phishing saluda ("Dear customer" -> "Hola"), y el modelo aprendía
# que "Hola, buenos días" ya era señal de phishing. Así el saludo deja de diferenciar las clases.
SALUDOS = {
    "es": ["Hola", "Hola, buenos días", "Buenos días", "Buenas tardes", "Buenas noches", "Hola a todos",
           "Hola equipo", "Estimados estudiantes", "Estimado profesor", "Hola Juan", "Hola María",
           "Cordial saludo", "Buen día", "Qué tal"],
    "en": ["Hi", "Hello", "Hi all", "Hello team", "Good morning", "Good afternoon", "Hey", "Hi John",
           "Hello Mary", "Dear all", "Dear professor", "Greetings"],
}
DESPEDIDAS = {
    "es": ["Saludos", "Gracias", "Muchas gracias", "Cordialmente", "Un abrazo", "Quedo atento",
           "Quedo atenta", "Feliz día", "Atentamente"],
    "en": ["Regards", "Best regards", "Thanks", "Thank you", "Best", "Cheers", "Sincerely", "Kind regards"],
}


def add_greetings(df, seed, p=0.5):
    """Antepone un saludo y agrega una despedida al azar (cada uno con probabilidad p)."""
    rng = np.random.default_rng(seed)
    out = []
    for text, src in zip(df.model_text, df.source):
        lang = "es" if str(src).endswith("_es") else "en"
        if rng.random() < p:
            text = for_model(rng.choice(SALUDOS[lang])) + " " + text
        if rng.random() < p:
            text = text + " " + for_model(rng.choice(DESPEDIDAS[lang]))
        out.append(text)
    df = df.copy()
    df["model_text"] = out
    return df


def greeting_false_alarms(model, te, thr):
    """De los correos legítimos de prueba que el modelo acierta, cuántos pasan a phishing al saludar."""
    legit = te[te.label == 0]
    base = model.predict(X(legit), batch_size=256, verbose=0).ravel() < thr
    res = {}
    for lang, greet in (("es", "hola buenos días"), ("en", "hello good morning")):
        mask = legit.source.str.endswith("_es").values == (lang == "es")
        sub = legit[mask & base]
        if not len(sub):
            continue
        pg = model.predict(np.array([greet + " " + t for t in sub.model_text], dtype=object),
                           batch_size=256, verbose=0).ravel()
        res[lang] = {"correos": int(len(sub)), "pasan_a_phishing": round(float((pg >= thr).mean()), 4)}
    return res


def X(d):
    return np.array(d.model_text.tolist(), dtype=object)


def build_model(train_texts, vocab=VOCAB):
    vectorize = layers.TextVectorization(
        max_tokens=vocab, output_sequence_length=MAX_LEN, standardize=None, name="vectorizar"
    )
    vectorize.adapt(train_texts)
    inp = keras.Input(shape=(1,), dtype="string", name="texto")
    x = vectorize(inp)
    x = layers.Embedding(vocab, EMB, mask_zero=True, name="embedding")(x)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(UNITS), name="bilstm")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="prob_phishing")(x)
    model = keras.Model(inp, out, name="shielddefence_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Recall(name="recall"), keras.metrics.Precision(name="precision")],
    )
    return model


def pick_threshold(y, p, min_precision=0.95):
    """Umbral que da el mayor recall manteniendo la precisión por encima del mínimo."""
    y = np.asarray(y)
    best_t, best_key = 0.5, (-1.0, -1.0)
    for t in np.arange(0.20, 0.81, 0.01):
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        prec = tp / max(pred.sum(), 1)
        rec = tp / max((y == 1).sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if prec >= min_precision and (rec, f1) > best_key:
            best_t, best_key = float(t), (rec, f1)
    return round(best_t, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--data", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--vocab", type=int, default=VOCAB)
    ap.add_argument("--min_precision", type=float, default=0.95)
    ap.add_argument("--sin_saludos", action="store_true", help="no agregar saludos al entrenar")
    a = ap.parse_args()
    keras.utils.set_random_seed(42)

    D = Path(a.data)
    tr, va, te = (pd.read_csv(D / f"{s}.csv") for s in ("train", "val", "test"))
    for d in (tr, va, te):
        d["model_text"] = d["model_text"].fillna("").astype(str)
    if not a.sin_saludos:
        tr, va = add_greetings(tr, 1), add_greetings(va, 2)

    model = build_model(X(tr), a.vocab)
    model.summary()
    hist = model.fit(
        X(tr), tr.label.values, validation_data=(X(va), va.label.values),
        epochs=a.epochs, batch_size=a.batch, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
    )

    pv = model.predict(X(va), batch_size=256, verbose=0).ravel()
    thr = pick_threshold(va.label.values, pv, a.min_precision)
    pt = model.predict(X(te), batch_size=256, verbose=0).ravel()
    res = {"umbral": thr, "precision_minima": a.min_precision, "saludos_al_entrenar": not a.sin_saludos,
           "test": report(te.label, pt, te.source, thr),
           "falsas_alarmas_por_saludo": greeting_false_alarms(model, te, thr)}

    es_path = ROOT / "data" / "spanish_eval.csv"
    if es_path.exists():
        es = pd.read_csv(es_path)
        pe = model.predict(np.array(es.text.map(for_model).tolist(), dtype=object), verbose=0).ravel()
        res["spanish"] = report(es.label, pe, threshold=thr)
        res["spanish_detalle"] = [
            {"nota": n, "real": int(l), "prob": round(float(p), 3)} for n, l, p in zip(es.note, es.label, pe)
        ]
    res["historial"] = {k: [round(float(x), 4) for x in v] for k, v in hist.history.items()}
    res["epocas"] = len(hist.history["loss"])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "lstm.keras")
    json.dump(res, open(out / "lstm_config.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k not in ("historial", "spanish_detalle")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
