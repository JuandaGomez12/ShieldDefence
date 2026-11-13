"""Ajuste fino del LLM explicador con LoRA.

Entrena solo una parte pequeña del modelo (adaptadores LoRA) para que aprenda a
explicar en español el resultado de la LSTM con el formato de ShieldDefence.
La pérdida se calcula solo sobre la respuesta del asistente, no sobre el correo.

Uso en Colab (GPU):
  python training/train_lora.py --base Qwen/Qwen2.5-0.5B-Instruct \
      --data data/explanations --out models/explicador --epochs 2 \
      --push_to TU_USUARIO/shielddefence-explicador
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          set_seed)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tok, max_len):
        self.items = []
        skipped = 0
        for r in rows:
            msgs = r["messages"]
            # Se arma el texto con la plantilla de chat y luego se tokeniza: así funciona igual
            # en transformers 4 y 5.
            prompt_txt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)
            full_txt = tok.apply_chat_template(msgs, tokenize=False)
            prompt_ids = tok(prompt_txt, add_special_tokens=False)["input_ids"]
            full_ids = tok(full_txt, add_special_tokens=False)["input_ids"]
            if full_ids[:len(prompt_ids)] != prompt_ids:
                skipped += 1
                continue
            if len(full_ids) > max_len:
                # Se recorta el correo (parte del prompt), nunca la respuesta.
                answer = full_ids[len(prompt_ids):]
                keep = max_len - len(answer)
                if keep < 64:
                    skipped += 1
                    continue
                prompt_ids = prompt_ids[:keep // 2] + prompt_ids[-(keep - keep // 2):]
                full_ids = prompt_ids + answer
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            self.items.append((full_ids, labels))
        print(f"{len(self.items)} ejemplos ({skipped} descartados)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(pad_id):
    def collate(batch):
        n = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for k, (x, y) in enumerate(batch):
            ids[k, :len(x)] = torch.tensor(x)
            lab[k, :len(y)] = torch.tensor(y)
            att[k, :len(x)] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="data/explanations")
    ap.add_argument("--out", default="models/explicador")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--push_to", default="")
    a = ap.parse_args()
    set_seed(42)

    gpu = torch.cuda.is_available()
    bf16 = gpu and torch.cuda.is_bf16_supported()
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        # En GPU sin bf16 (como la T4 de Colab) el modelo base se carga en fp32 y se entrena con
        # precisión mixta fp16; cargarlo directamente en fp16 hace fallar el escalado de gradientes.
        a.base, dtype=torch.bfloat16 if bf16 else torch.float32)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    d = Path(a.data)
    train = ChatDataset(load_jsonl(d / "train.jsonl"), tok, a.max_len)
    val = ChatDataset(load_jsonl(d / "val.jsonl"), tok, a.max_len)

    steps_per_epoch = math.ceil(len(train) / (a.batch * a.grad_accum))
    total_steps = a.max_steps if a.max_steps > 0 else math.ceil(steps_per_epoch * a.epochs)
    args = TrainingArguments(
        output_dir=str(Path(a.out) / "checkpoints"),
        num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr,
        lr_scheduler_type="cosine", warmup_steps=max(1, int(0.05 * total_steps)), weight_decay=0.0,
        logging_steps=10, eval_strategy="epoch" if a.max_steps < 0 else "no", save_strategy="no",
        bf16=bf16, fp16=gpu and not bf16, gradient_checkpointing=gpu,
        report_to=[], remove_unused_columns=False, dataloader_num_workers=0,
    )
    if gpu:
        model.enable_input_require_grads()
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                      data_collator=make_collator(tok.pad_token_id))
    trainer.train()
    ev = trainer.evaluate()
    print("Pérdida en validación:", ev.get("eval_loss"), "perplejidad:", math.exp(min(ev.get("eval_loss", 20), 20)))

    # Se fusiona LoRA con el modelo base para servirlo sin peft.
    out = Path(a.out)
    model.save_pretrained(out / "adaptador_lora")
    merged = model.merge_and_unload()
    merged = merged.to(torch.bfloat16)  # la mitad de tamaño; el servidor lo carga en fp32
    merged.save_pretrained(out / "fusionado", safe_serialization=True)
    tok.save_pretrained(out / "fusionado")
    json.dump({"base": a.base, "eval_loss": ev.get("eval_loss"), "ejemplos_train": len(train),
               "epocas": a.epochs, "rank": a.rank}, open(out / "lora_info.json", "w"), indent=2)
    print("Guardado en", out)

    if a.push_to:
        merged.push_to_hub(a.push_to, private=False, token=os.getenv("HF_TOKEN"))
        tok.push_to_hub(a.push_to, token=os.getenv("HF_TOKEN"))
        print("Subido a https://huggingface.co/" + a.push_to)


if __name__ == "__main__":
    main()
