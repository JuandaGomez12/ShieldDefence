"""API de ShieldDefence (FastAPI) y servidor de la interfaz.

  GET  /                 interfaz tipo chat
  GET  /api/estado       estado de los modelos
  POST /api/analizar     {"texto": "..."}  -> análisis del correo
  POST /api/analizar-archivo   (imagen o PDF) -> OCR + análisis
"""
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app import ocr  # noqa: E402
from app.pipeline import ShieldDefence  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
MAX_UPLOAD = 8 * 1024 * 1024
FRONT = ROOT / "frontend"

app = FastAPI(title="ShieldDefence", version="1.0")
shield = ShieldDefence()


class Entrada(BaseModel):
    texto: str = Field(..., min_length=1, max_length=20000)


@app.get("/api/estado")
def estado():
    return shield.status()


@app.post("/api/analizar")
def analizar(entrada: Entrada):
    texto = entrada.texto.strip()
    if len(texto) < 15:
        raise HTTPException(400, "El texto es muy corto para analizarlo como correo.")
    return shield.analyze(texto)


@app.post("/api/analizar-archivo")
async def analizar_archivo(archivo: UploadFile = File(...)):
    data = await archivo.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "El archivo pesa más de 8 MB.")
    tipo = (archivo.content_type or "").lower()
    if not (tipo.startswith("image/") or tipo == "application/pdf" or (archivo.filename or "").lower().endswith(".pdf")):
        raise HTTPException(415, "Sube una imagen (PNG o JPG) o un PDF.")
    try:
        texto = ocr.extract(archivo.filename, tipo, data).strip()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"No se pudo leer el archivo: {e}") from e
    if len(texto) < 15:
        raise HTTPException(422, "No se encontró texto suficiente en el archivo.")
    res = shield.analyze(texto)
    res["texto_extraido"] = texto[:4000]
    return res


@app.get("/")
def index():
    return FileResponse(FRONT / "index.html")


app.mount("/", StaticFiles(directory=FRONT), name="front")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
