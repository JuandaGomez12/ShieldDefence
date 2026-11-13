"""La BiLSTM de ShieldDefence ejecutada con NumPy.

El modelo se entrena con Keras (training/train_lstm.py) y después se exportan
sus pesos (training/export_lstm.py). En el servidor no hace falta TensorFlow:
este archivo hace exactamente las mismas cuentas que las capas de Keras
(Embedding -> BiLSTM -> Dense -> Dense sigmoide).
"""
import json
from pathlib import Path

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyBiLSTM:
    def __init__(self, folder):
        folder = Path(folder)
        w = np.load(folder / "lstm_pesos.npz")
        meta = json.load(open(folder / "lstm_vocab.json", encoding="utf-8"))
        self.vocab = {tok: i for i, tok in enumerate(meta["vocab"])}
        self.max_len = meta["max_len"]
        self.emb = w["embedding"]
        self.fw = (w["fw_kernel"], w["fw_recurrent"], w["fw_bias"])
        self.bw = (w["bw_kernel"], w["bw_recurrent"], w["bw_bias"])
        self.d1 = (w["dense_kernel"], w["dense_bias"])
        self.out = (w["out_kernel"], w["out_bias"])
        self.units = self.fw[1].shape[0]

    def tokens(self, text):
        ids = [self.vocab.get(t, 1) for t in text.split()[: self.max_len]]
        return [i for i in ids if i != 0]

    def _run(self, x, params):
        k, r, b = params
        u = self.units
        h = np.zeros(u, dtype=np.float32)
        c = np.zeros(u, dtype=np.float32)
        z_in = x @ k + b                      # se calcula la parte de la entrada de una vez
        for t in range(x.shape[0]):
            z = z_in[t] + h @ r
            i, f, g, o = _sigmoid(z[:u]), _sigmoid(z[u:2 * u]), np.tanh(z[2 * u:3 * u]), _sigmoid(z[3 * u:])
            c = f * c + i * g
            h = o * np.tanh(c)
        return h

    def predict_ids(self, ids):
        if not ids:
            ids = [1]
        x = self.emb[np.asarray(ids)]
        h = np.concatenate([self._run(x, self.fw), self._run(x[::-1], self.bw)])
        h = np.maximum(h @ self.d1[0] + self.d1[1], 0.0)
        return float(_sigmoid(h @ self.out[0] + self.out[1])[0])

    def predict(self, model_text: str) -> float:
        return self.predict_ids(self.tokens(model_text))
