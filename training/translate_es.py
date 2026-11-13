"""Traduce parte del dataset al español para que la LSTM aprenda los dos idiomas.

Se traduce una fracción de cada partición (train, val y test por separado, para
que ningún correo traducido quede en una partición distinta a la de su original)
con el modelo Helsinki-NLP/opus-mt-en-es (MarianMT). Los enlaces y correos se
protegen para que el traductor no los cambie.

Uso en Colab (GPU):  python training/translate_es.py --frac 0.5
Prueba sin modelo:   python training/translate_es.py --dry-run --limit 20
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shield_core.preprocess import for_model  # noqa: E402

PROTECT = re.compile(r"(https?://[^\s<>\"')\]]+|www\.[^\s<>\"')\]]+|[\w.+-]+@[\w-]+\.[\w.-]+)")
MAX_CHARS = 1500     # se traduce el asunto y el comienzo del correo
CHUNK = 350          # tamaño máximo de cada fragmento que recibe el traductor


def protect(text):
    keep = []

    def sub(m):
        keep.append(m.group(0))
        return f"ZXQ{len(keep) - 1}QXZ"
    return PROTECT.sub(sub, text), keep


def restore(text, keep):
    def sub(m):
        i = int(m.group(1))
        return keep[i] if i < len(keep) else ""
    return re.sub(r"ZXQ\s*(\d+)\s*QXZ", sub, text)


def chunks(text):
    """Divide en fragmentos por oración sin pasar de CHUNK caracteres. Respeta los saltos de línea."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            out.append(("\n", False))
            continue
        cur = ""
        for sent in re.split(r"(?<=[.!?])\s+", line):
            while len(sent) > CHUNK:
                out.append((sent[:CHUNK], True))
                sent = sent[CHUNK:]
            if len(cur) + len(sent) + 1 > CHUNK and cur:
                out.append((cur, True))
                cur = sent
            else:
                cur = (cur + " " + sent).strip()
        if cur:
            out.append((cur, True))
        out.append(("\n", False))
    return out


class Translator:
    def __init__(self, model_id, batch, dry_run=False):
        self.dry = dry_run
        self.batch = batch
        if dry_run:
            return
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = MarianTokenizer.from_pretrained(model_id)
        self.model = MarianMTModel.from_pretrained(model_id).to(self.dev).eval()
        if self.dev == "cuda":
            self.model = self.model.half()

    def __call__(self, segs):
        if self.dry:
            return [f"[es] {s}" for s in segs]
        res = []
        for i in range(0, len(segs), self.batch):
            b = segs[i:i + self.batch]
            enc = self.tok(b, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.dev)
            with self.torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=256, num_beams=1)
            res += self.tok.batch_decode(out, skip_special_tokens=True)
        return res


def translate_texts(texts, tr):
    """Traduce una lista de correos juntando todos los fragmentos en lotes grandes."""
    plans, segs = [], []
    for t in texts:
        t, keep = protect(str(t)[:MAX_CHARS])
        plan = []
        for c, needs in chunks(t):
            if needs:
                plan.append(len(segs))
                segs.append(c)
            else:
                plan.append(c)
        plans.append((plan, keep))
    order = sorted(range(len(segs)), key=lambda i: len(segs[i]))   # lotes de largo parecido = más rápido
    done = tr([segs[i] for i in order])
    trans = [None] * len(segs)
    for i, t in zip(order, done):
        trans[i] = t
    out = []
    for plan, keep in plans:
        parts = [trans[p] if isinstance(p, int) else p for p in plan]
        txt = re.sub(r"[ \t]*\n[ \t]*", "\n", " ".join(parts)).strip()
        out.append(restore(txt, keep))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed_bilingue"))
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--model", default="Helsinki-NLP/opus-mt-en-es")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="solo para pruebas")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tr = Translator(a.model, a.batch, a.dry_run)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        df = pd.read_csv(Path(a.src) / f"{split}.csv")
        pick = df.groupby("source", group_keys=False).sample(frac=a.frac, random_state=11)
        if a.limit:
            pick = pick.head(a.limit)
        es = pick.copy()
        es["text"] = translate_texts(pick.text.tolist(), tr)
        es["model_text"] = es.text.map(for_model)
        es["source"] = es.source + "_es"
        both = pd.concat([df, es], ignore_index=True).sample(frac=1, random_state=3)
        both.to_csv(out / f"{split}.csv", index=False)
        print(f"{split}: {len(df)} originales + {len(es)} traducidos = {len(both)}")
    print("\nEjemplo traducido:\n", es.text.iloc[0][:500])


if __name__ == "__main__":
    main()
