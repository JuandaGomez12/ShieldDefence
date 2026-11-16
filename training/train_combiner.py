"""Entrena el combinador (BiLSTM + señales) y compara contra la BiLSTM sola.

1. Calcula la probabilidad de la BiLSTM (versión NumPy, la misma que usa la página) para validación.
2. Ajusta una regresión logística con esa probabilidad y las señales de las reglas.
3. Elige el umbral en validación (precisión mínima, igual que la LSTM).
4. Evalúa la BiLSTM sola y el combinador en prueba y en la prueba en español escrita a mano, por categoría,
   y lista los correos que fallan.

Uso: python training/train_combiner.py --data data/processed_bilingue --extra data/synthetic_es
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shield_core.combine import FEATURES, features, rule_points  # noqa: E402
from shield_core.lstm_numpy import NumpyBiLSTM  # noqa: E402
from shield_core.preprocess import clean_raw, for_model  # noqa: E402
from shield_core.signals import detect  # noqa: E402
from training.metrics import report  # noqa: E402
from training.metrics import pick_threshold  # noqa: E402


KERAS = None  # si existe models/lstm.keras se usa por lotes (mucho más rápido); da lo mismo que NumPy


def lstm_probs(model, texts, noise=0.0, seed=0):
    """Probabilidad de la BiLSTM. Con noise > 0 cambia esa fracción de palabras por desconocidas."""
    rng = np.random.default_rng(seed)
    mts = []
    for t in texts:
        mt = for_model(t)
        if noise:
            mt = " ".join("zqzq" if rng.random() < noise else w for w in mt.split())
        mts.append(mt)
    if KERAS is not None:
        return KERAS.predict(np.array(mts, dtype=object), batch_size=256, verbose=0).ravel()
    return np.array([model.predict(mt) for mt in mts])


def feats(texts, probs):
    return np.array([features(clean_raw(t), p, detect(clean_raw(t))) for t, p in zip(texts, probs)])


KEYWORDS = ["clave", "contraseña", "cuenta", "tarjeta", "bloqueada", "suspendida", "verificar", "urgente",
            "password", "account", "verify", "suspended", "bank", "banco", "premio", "pago", "datos"]
CYR = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "i": "і"}
LEET_OUT = {"a": "4", "e": "3", "o": "0", "i": "1", "s": "5"}


def obfuscate(text, rng):
    """Disfraza palabras clave de un correo como lo haría un atacante (para entrenar el combinador)."""
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if w.strip(".,:;!¡?¿").lower() in KEYWORDS]
    if not idx:
        idx = list(rng.choice(len(words), size=min(2, len(words)), replace=False))
    for i in idx:
        w = words[i]
        trick = rng.integers(5)
        if trick == 0:
            words[i] = " ".join(w)
        elif trick == 1:
            words[i] = rng.choice([".", "-", "*"]).join(w)
        elif trick == 2:
            words[i] = "".join(LEET_OUT.get(c, c) if rng.random() < 0.6 else c for c in w)
        elif trick == 3:
            words[i] = w[: len(w) // 2] + "\u200b" + w[len(w) // 2:]
        else:
            words[i] = "".join(CYR.get(c, c) if rng.random() < 0.5 else c for c in w)
    return " ".join(words)


class BoundedLogReg:
    """Regresión logística donde las señales de riesgo solo pueden subir la probabilidad (peso >= 0).

    Sin esta restricción, si en validación la LSTM ya acertaba un caso, la regresión podía darle peso negativo
    a una señal (por ejemplo, a "dominio que imita una marca") y un correo con un dominio falso terminaba
    con menos probabilidad de phishing. Los únicos pesos libres son el número de enlaces y "solo informa".
    """

    def __init__(self, C=1.0, free=("n_urls", "solo_informa"), zero=()):
        self.C, self.free, self.zero = C, set(free), set(zero)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        sw = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))  # clases balanceadas
        Xb = np.hstack([X, np.ones((len(X), 1))])

        def loss(w):
            z = Xb @ w
            l = np.sum(sw * (np.logaddexp(0, z) - y * z)) + 0.5 / self.C * np.sum(w[:-1] ** 2)
            g = Xb.T @ (sw * (1 / (1 + np.exp(-z)) - y))
            g[:-1] += w[:-1] / self.C
            return l, g

        bounds = [(0, 0) if f in self.zero else (None, None) if f in self.free else (0, None) for f in FEATURES] + [(None, None)]
        r = minimize(loss, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B", bounds=bounds)
        self.coef_, self.intercept_ = r.x[None, :-1], r.x[-1:]
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(np.asarray(X, float) @ self.coef_[0] + self.intercept_[0])))
        return np.column_stack([1 - p, p])


def evaluate(name, df, p, thr, show_errors=False):
    pred = p >= thr
    ok = pred == df.label.values.astype(bool)
    out = {"accuracy": round(float(ok.mean()), 4), "falsos_positivos": int((pred & (df.label.values == 0)).sum()),
           "phishing_que_se_escapa": int((~pred & (df.label.values == 1)).sum())}
    if "categoria" in df:
        out["por_categoria"] = {c: f"{int(ok[m].sum())}/{int(m.sum())}" for c in sorted(df.categoria.unique())
                                for m in [df.categoria.values == c]}
    if show_errors:
        out["errores"] = [f"[{'phishing' if l else 'legítimo'} -> {pp:.2f}] {n}"
                          for n, l, pp, o in zip(df.note, df.label, p, ok) if not o]
    print(f"\n== {name}: {json.dumps({k: v for k, v in out.items() if k != 'errores'}, ensure_ascii=False)}")
    for e in out.get("errores", []):
        print("   ", e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "processed_bilingue"))
    ap.add_argument("--extra", default="", help="carpeta con sintéticos (se usa su val.csv)")
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--min_precision", type=float, default=0.95)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--ciega", action="store_true",
                    help="evaluar también data/spanish_blind.csv (prueba ciega: usarla solo al final, una vez)")
    ap.add_argument("--max_legit_piso", type=float, default=0.005,
                    help="fracción máxima de correos legítimos de validación que puede activar la red de seguridad")
    ap.add_argument("--blind", default="", help="otra prueba ciega para evaluar (ruta a un csv)")
    ap.add_argument("--sin", default="", help="variables que no se usan (peso 0), separadas por coma")
    ap.add_argument("--solo_numpy", action="store_true", help="no usar Keras aunque exista lstm.keras")
    ap.add_argument("--disfrazados", type=float, default=0.2,
                    help="fracción de correos de phishing de validación que se agregan disfrazados")
    ap.add_argument("--ruido", type=float, default=0.35,
                    help="fracción de palabras desconocidas al calcular la LSTM en validación. En validación la LSTM "
                         "casi no se equivoca y el combinador aprendería a confiar solo en ella; con ruido se simula "
                         "un correo con vocabulario nuevo y aprende cuándo apoyarse en las reglas")
    a = ap.parse_args()
    M = Path(a.models)
    model = NumpyBiLSTM(M)
    global KERAS
    if (M / "lstm.keras").exists() and not a.solo_numpy:
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import keras
        KERAS = keras.models.load_model(M / "lstm.keras")
    lstm_thr = json.load(open(M / "lstm_config.json"))["umbral"]

    D = Path(a.data)
    va, te = pd.read_csv(D / "val.csv"), pd.read_csv(D / "test.csv")
    if a.extra:
        va = pd.concat([va, pd.read_csv(Path(a.extra) / "val.csv")], ignore_index=True)
    es = pd.read_csv(ROOT / "data" / "spanish_eval.csv")
    for d in (va, te, es):
        d["text"] = d["text"].fillna("").astype(str)

    print("Calculando la BiLSTM en validación, prueba y español...", flush=True)
    pv, pt, pe = lstm_probs(model, va.text), lstm_probs(model, te.text), lstm_probs(model, es.text)
    pv_ruido = lstm_probs(model, va.text, a.ruido, 1)
    Xv, Xt, Xe = feats(va.text, pv), feats(te.text, pt), feats(es.text, pe)
    Xv_ruido = feats(va.text, pv_ruido)
    # Copias disfrazadas de una parte del phishing ("c l a v e", "cl4ve"...). Los legítimos no se disfrazan
    # porque en la vida real no escriben así.
    rng = np.random.default_rng(5)
    ph = va[va.label == 1].sample(frac=a.disfrazados, random_state=5)
    ph_txt = [obfuscate(t, rng) for t in ph.text]
    X_disf = feats(ph_txt, lstm_probs(model, ph_txt))

    lr = BoundedLogReg(C=a.C, zero=a.sin.split(",") if a.sin else ())
    # Se entrena con la validación limpia y con la versión con palabras desconocidas.
    lr.fit(np.vstack([Xv, Xv_ruido, X_disf]), np.concatenate([va.label.values, va.label.values, np.ones(len(X_disf))]))
    cv = lr.predict_proba(Xv)[:, 1]
    # Umbral que maximiza la exactitud en validación (con y sin palabras desconocidas): así pesan igual
    # las falsas alarmas y el phishing que se escapa.
    cv_ruido = lr.predict_proba(Xv_ruido)[:, 1]
    yy, pp = np.concatenate([va.label.values] * 2), np.concatenate([cv, cv_ruido])
    grid = np.arange(0.05, 0.96, 0.01)
    thr = round(float(grid[np.argmax([((pp >= t) == yy).mean() for t in grid])]), 2)
    # Red de seguridad: el menor puntaje de reglas que casi ningún legítimo de validación alcanza.
    pts_v = np.array([rule_points(detect(clean_raw(t))) for t in va.text])
    legit_pts = np.sort(pts_v[va.label.values == 0])
    piso = None
    for cand in range(40, 200, 5):
        if (legit_pts >= cand).mean() <= a.max_legit_piso:
            piso = cand
            break
    print(f"Red de seguridad: {piso} puntos de reglas (legítimos de validación que la activan: "
          f"{(legit_pts >= piso).mean():.2%}; phishing de validación: {(pts_v[va.label.values == 1] >= piso).mean():.1%})")

    def con_piso(probs, texts):
        pts = np.array([rule_points(detect(clean_raw(t))) for t in texts])
        return np.where(pts >= piso, np.maximum(probs, 0.9), probs)

    ct, ce = con_piso(lr.predict_proba(Xt)[:, 1], te.text), con_piso(lr.predict_proba(Xe)[:, 1], es.text)

    pesos = dict(zip(FEATURES, [round(float(w), 4) for w in lr.coef_[0]]))
    print("Pesos del combinador:", json.dumps(pesos, ensure_ascii=False))
    print("Sesgo:", round(float(lr.intercept_[0]), 4), "| umbral:", thr)

    res = {"lstm_sola": {"umbral": lstm_thr, "test": report(te.label, pt, te.source, lstm_thr),
                         "espanol": evaluate("BiLSTM sola, español", es, pe, lstm_thr, True)},
           "combinador": {"umbral": thr, "test": report(te.label, ct, te.source, thr),
                          "espanol": evaluate("Combinador, español", es, ce, thr, True)}}
    # Phishing de la prueba en español, disfrazado: cuánto se sigue detectando.
    rng2 = np.random.default_rng(11)
    ph_es = es[es.label == 1]
    disf = [obfuscate(t, rng2) for t in ph_es.text]
    pd_l = lstm_probs(model, disf)
    pd_c = con_piso(lr.predict_proba(feats(disf, pd_l))[:, 1], disf)
    res["phishing_disfrazado_es"] = {"correos": len(disf), "lstm_sola": round(float((pd_l >= lstm_thr).mean()), 4),
                                     "combinador": round(float((pd_c >= thr).mean()), 4)}
    print("\nPhishing en español disfrazado detectado:", json.dumps(res["phishing_disfrazado_es"]))
    if a.ciega:
        bl = pd.read_csv(ROOT / "data" / "spanish_blind.csv")
        pb = lstm_probs(model, bl.text)
        cb = con_piso(lr.predict_proba(feats(bl.text, pb))[:, 1], bl.text)
        res["ciega"] = {"lstm_sola": evaluate("PRUEBA CIEGA, BiLSTM sola", bl, pb, lstm_thr, True),
                        "combinador": evaluate("PRUEBA CIEGA, combinador", bl, cb, thr, True)}
    if a.blind:
        b2 = pd.read_csv(a.blind)
        pb2 = lstm_probs(model, b2.text)
        cb2 = con_piso(lr.predict_proba(feats(b2.text, pb2))[:, 1], b2.text)
        res["ciega2"] = {"lstm_sola": evaluate(f"{a.blind}, BiLSTM sola", b2, pb2, lstm_thr, True),
                         "combinador": evaluate(f"{a.blind}, combinador", b2, cb2, thr, True)}
    print("\nPrueba general  BiLSTM sola:", res["lstm_sola"]["test"]["accuracy"],
          "| combinador:", res["combinador"]["test"]["accuracy"])
    json.dump({"features": FEATURES, "pesos": [float(w) for w in lr.coef_[0]], "sesgo": float(lr.intercept_[0]),
               "umbral": thr, "piso_reglas": piso, "piso_prob": 0.9}, open(M / "combinador.json", "w"), indent=2)
    json.dump(res, open(M / "combinador_eval.json", "w"), indent=2, ensure_ascii=False)
    print("Guardado en", M / "combinador.json")


if __name__ == "__main__":
    main()
