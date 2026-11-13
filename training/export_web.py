"""Exporta la BiLSTM al formato que lee la versión web (web/js/shield-core.js).

Crea dos archivos:
  lstm_web.json  vocabulario, umbral y dónde está cada tensor dentro del .bin
  lstm_web.bin   los pesos (embeddings en float16 para que pese la mitad, el resto en float32)

Uso: python training/export_web.py --models models --out web/model [--push_to USUARIO/shielddefence-lstm]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--out", default=str(ROOT / "web" / "model"))
    ap.add_argument("--push_to", default="")
    a = ap.parse_args()
    M, out = Path(a.models), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w = np.load(M / "lstm_pesos.npz")
    vocab = json.load(open(M / "lstm_vocab.json", encoding="utf-8"))
    cfg = json.load(open(M / "lstm_config.json", encoding="utf-8"))

    tensors, blobs, offset = {}, [], 0
    for name in w.files:
        arr = w[name]
        arr = arr.astype(np.float16) if name == "embedding" else arr.astype(np.float32)
        raw = arr.tobytes()
        pad = (-offset) % 4  # Float32Array necesita posiciones múltiplos de 4
        if pad:
            blobs.append(b"\0" * pad)
            offset += pad
        tensors[name] = {"dtype": str(arr.dtype), "shape": list(arr.shape), "offset": offset, "size": int(arr.size)}
        blobs.append(raw)
        offset += len(raw)
    (out / "lstm_web.bin").write_bytes(b"".join(blobs))
    meta = {"vocab": vocab["vocab"], "max_len": vocab["max_len"], "umbral": cfg["umbral"],
            "metricas": {"test": cfg.get("test"), "espanol": cfg.get("spanish")}, "tensors": tensors}
    json.dump(meta, open(out / "lstm_web.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"Exportado en {out}: {offset / 2**20:.1f} MB de pesos")

    if a.push_to:
        from huggingface_hub import HfApi
        api = HfApi(token=os.getenv("HF_TOKEN"))
        api.create_repo(a.push_to, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=a.push_to, commit_message="BiLSTM para la versión web")
        for f in ("lstm_pesos.npz", "lstm_vocab.json", "lstm_config.json"):
            api.upload_file(path_or_fileobj=str(M / f), path_in_repo=f"numpy/{f}", repo_id=a.push_to)
        print("Subido a https://huggingface.co/" + a.push_to)


if __name__ == "__main__":
    main()
