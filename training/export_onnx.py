"""Exporta el LLM explicador a ONNX para correrlo en el navegador con Transformers.js.

1. Exporta el modelo fusionado (LoRA ya aplicado) a ONNX con optimum.
2. Lo cuantiza: las multiplicaciones de matrices a 4 bits (MatMulNBits) y la
   tabla de embeddings a 8 bits. Así baja de ~2 GB a unos 400 MB.
3. Deja la estructura que espera Transformers.js: config y tokenizador en la raíz
   y el modelo en onnx/model_q4.onnx.
4. Opcional: lo sube a Hugging Face.

Uso: python training/export_onnx.py --model models/explicador/fusionado \
         --out models/explicador_onnx --push_to USUARIO/shielddefence-explicador-onnx
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def quantize(src: Path, dst: Path, block: int):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    tmp = dst.with_name("tmp_q4.onnx")
    model = onnx.load(str(src), load_external_data=True)
    q = MatMulNBitsQuantizer(model, block_size=block, is_symmetric=True, accuracy_level=4)
    q.process()
    q.model.save_model_to_file(str(tmp), use_external_data_format=False)
    # La tabla de embeddings (Gather) pasa a 8 bits; las MatMul ya quedaron en 4 bits.
    quantize_dynamic(str(tmp), str(dst), op_types_to_quantize=["Gather"], weight_type=QuantType.QUInt8,
                     extra_options={"EnableSubgraph": True})
    tmp.unlink()
    for f in dst.parent.glob("tmp_q4.onnx*"):
        f.unlink()


def untie(src, dst):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32)
    if getattr(model.config, "tie_word_embeddings", False):
        emb = model.get_input_embeddings().weight
        model.lm_head.weight = torch.nn.Parameter(emb.detach().clone())
        model.config.tie_word_embeddings = False
    shutil.rmtree(dst, ignore_errors=True)
    model.save_pretrained(dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/explicador/fusionado")
    ap.add_argument("--out", default="models/explicador_onnx")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--push_to", default="")
    a = ap.parse_args()
    out = Path(a.out)
    raw = out.with_name(out.name + "_fp32")
    shutil.rmtree(raw, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    # Qwen comparte la matriz de embeddings con la capa de salida (lm_head). En ONNX eso hace que
    # lm_head no se pueda cuantizar, así que se separan antes de exportar.
    untied = out.with_name(out.name + "_untied")
    untie(a.model, untied)
    run([sys.executable, "-m", "optimum.commands.optimum_cli", "export", "onnx", "--model", str(untied),
         "--task", "text-generation-with-past", str(raw)])
    shutil.rmtree(untied, ignore_errors=True)

    (out / "onnx").mkdir(parents=True)
    for f in raw.iterdir():
        if not f.name.startswith("model.onnx"):
            shutil.copy(f, out / f.name)
    # Transformers.js lee la plantilla de chat desde tokenizer_config.json.
    tpl = out / "chat_template.jinja"
    cfg_path = out / "tokenizer_config.json"
    if tpl.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg.setdefault("chat_template", tpl.read_text(encoding="utf-8"))
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Cuantizando a 4 bits...", flush=True)
    quantize(raw / "model.onnx", out / "onnx" / "model_q4.onnx", a.block)
    size = sum(f.stat().st_size for f in (out / "onnx").iterdir()) / 2**20
    print(f"Modelo ONNX listo en {out} ({size:.0f} MB)")

    if a.push_to:
        from huggingface_hub import HfApi
        api = HfApi(token=os.getenv("HF_TOKEN"))
        api.create_repo(a.push_to, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=a.push_to, commit_message="Modelo ONNX para el navegador")
        print("Subido a https://huggingface.co/" + a.push_to)


if __name__ == "__main__":
    main()
