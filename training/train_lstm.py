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
from training.metrics import pick_threshold, report  # noqa: E402,F401

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


def add_unknown_words(df, rate, seed):
    """Cambia al azar una parte de las palabras por palabras inventadas (que quedan fuera del vocabulario).

    En español real aparecen muchas palabras que el modelo nunca vio ("homologación", "receta"...), y la
    BiLSTM había aprendido a asociar las palabras desconocidas con phishing. Con esto ve palabras
    desconocidas en las dos clases por igual y deja de usarlas como pista.
    """
    rng = np.random.default_rng(seed)
    out = []
    for text in df.model_text:
        words = text.split()
        mask = rng.random(len(words)) < rate
        out.append(" ".join(f"zq{rng.integers(1e9):x}" if m else w for w, m in zip(words, mask)))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--data", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--vocab", type=int, default=VOCAB)
    ap.add_argument("--min_precision", type=float, default=0.95)
    ap.add_argument("--sin_saludos", action="store_true", help="no agregar saludos al entrenar")
    ap.add_argument("--desconocidas", type=float, default=0.12,
                    help="fracción de palabras que se cambian por desconocidas al entrenar (0 para no hacerlo)")
    ap.add_argument("--extra", default="", help="carpeta con correos sintéticos (train.csv y val.csv) para sumar")
    a = ap.parse_args()
    keras.utils.set_random_seed(42)

    D = Path(a.data)
    tr, va, te = (pd.read_csv(D / f"{s}.csv") for s in ("train", "val", "test"))
    if a.extra:
        E = Path(a.extra)
        cols = ["text", "model_text", "label", "source"]
        tr = pd.concat([tr, pd.read_csv(E / "train.csv")[cols]], ignore_index=True).sample(frac=1, random_state=42)
        va = pd.concat([va, pd.read_csv(E / "val.csv")[cols]], ignore_index=True)
        print(f"Con sintéticos: {len(tr)} de entrenamiento, {len(va)} de validación")
    for d in (tr, va, te):
        d["model_text"] = d["model_text"].fillna("").astype(str)
    if not a.sin_saludos:
        tr, va = add_greetings(tr, 1), add_greetings(va, 2)

    # El vocabulario se arma con el texto original; el ruido se agrega después, así las palabras
    # inventadas quedan fuera del vocabulario.
    model = build_model(X(tr), a.vocab)
    if a.desconocidas > 0:
        tr = add_unknown_words(tr, a.desconocidas, 3)
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
           "palabras_desconocidas_al_entrenar": a.desconocidas,
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
