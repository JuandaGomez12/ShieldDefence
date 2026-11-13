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


def pick_threshold(y, p, min_precision=0.85):
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
    a = ap.parse_args()
    keras.utils.set_random_seed(42)

    D = Path(a.data)
    tr, va, te = (pd.read_csv(D / f"{s}.csv") for s in ("train", "val", "test"))
    for d in (tr, va, te):
        d["model_text"] = d["model_text"].fillna("").astype(str)
    X = lambda d: np.array(d.model_text.tolist(), dtype=object)  # noqa: E731

    model = build_model(X(tr), a.vocab)
    model.summary()
    hist = model.fit(
        X(tr), tr.label.values, validation_data=(X(va), va.label.values),
        epochs=a.epochs, batch_size=a.batch, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
    )

    pv = model.predict(X(va), batch_size=256, verbose=0).ravel()
    thr = pick_threshold(va.label.values, pv)
    pt = model.predict(X(te), batch_size=256, verbose=0).ravel()
    res = {"umbral": thr, "test": report(te.label, pt, te.source, thr)}

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
