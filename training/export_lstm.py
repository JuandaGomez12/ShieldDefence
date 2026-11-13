"""Exporta la LSTM de Keras a NumPy y comprueba que da los mismos resultados.

Uso: python training/export_lstm.py [--models models] [--check 300]
Crea models/lstm_pesos.npz y models/lstm_vocab.json.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shield_core.lstm_numpy import NumpyBiLSTM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--data", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--check", type=int, default=300)
    a = ap.parse_args()
    M = Path(a.models)
    model = keras.models.load_model(M / "lstm.keras")
    vec = model.get_layer("vectorizar")
    bil = model.get_layer("bilstm")
    dense = [l for l in model.layers if isinstance(l, keras.layers.Dense)]
    fk, fr, fb = bil.forward_layer.get_weights()
    bk, br, bb = bil.backward_layer.get_weights()
    np.savez_compressed(
        M / "lstm_pesos.npz",
        embedding=model.get_layer("embedding").get_weights()[0],
        fw_kernel=fk, fw_recurrent=fr, fw_bias=fb, bw_kernel=bk, bw_recurrent=br, bw_bias=bb,
        dense_kernel=dense[0].get_weights()[0], dense_bias=dense[0].get_weights()[1],
        out_kernel=dense[1].get_weights()[0], out_bias=dense[1].get_weights()[1],
    )
    max_len = vec.get_config()["output_sequence_length"]
    json.dump({"vocab": vec.get_vocabulary(), "max_len": max_len},
              open(M / "lstm_vocab.json", "w", encoding="utf-8"), ensure_ascii=False)

    # Comprobación: mismas probabilidades que Keras en correos del conjunto de prueba.
    test = pd.read_csv(Path(a.data) / "test.csv").sample(n=a.check, random_state=1)
    texts = test.model_text.fillna("").astype(str).tolist()
    pk = model.predict(np.array(texts, dtype=object), verbose=0).ravel()
    npm = NumpyBiLSTM(M)
    pn = np.array([npm.predict(t) for t in texts])
    diff = np.abs(pk - pn).max()
    print(f"Diferencia máxima Keras vs NumPy en {a.check} correos: {diff:.2e}")
    assert diff < 1e-3, "La versión NumPy no coincide con Keras"
    print("Exportado en", M)


if __name__ == "__main__":
    main()
