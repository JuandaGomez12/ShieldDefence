"""Sube ShieldDefence a un Space de Hugging Face (Docker).

Uso:  HF_TOKEN=hf_xxx python deploy/push_space.py --space USUARIO/shielddefence \
          --llm USUARIO/shielddefence-explicador

Solo sube lo que el servidor necesita: el código, la interfaz y la LSTM
exportada a NumPy. El LLM se descarga en el Space desde --llm.
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [
    "Dockerfile", "README.md", "requirements.txt",
    "app/*.py", "shield_core/*.py", "frontend/*",
    "models/lstm_pesos.npz", "models/lstm_vocab.json", "models/lstm_config.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--llm", default="")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    for f in ("models/lstm_pesos.npz", "models/lstm_vocab.json", "models/lstm_config.json"):
        assert (ROOT / f).exists(), f"Falta {f}: entrena y exporta la LSTM primero."

    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(a.space, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
    if a.llm:
        api.add_space_variable(a.space, "SHIELD_LLM_MODEL", a.llm)
    api.upload_folder(folder_path=str(ROOT), repo_id=a.space, repo_type="space",
                      allow_patterns=INCLUDE, commit_message="Despliegue de ShieldDefence")
    print(f"Listo: https://huggingface.co/spaces/{a.space}")
    print("La primera vez el Space tarda unos minutos en construir la imagen y descargar el LLM.")


if __name__ == "__main__":
    main()
