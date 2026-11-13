"""Saca el texto de un pantallazo o de un PDF del correo."""
import io
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

MAX_PDF_PAGES = 3


def _ocr_image(img: Image.Image) -> str:
    import pytesseract
    img = ImageOps.exif_transpose(img).convert("L")
    if img.width < 1400:  # los pantallazos pequeños se leen mejor ampliados
        f = 1400 / img.width
        img = img.resize((int(img.width * f), int(img.height * f)))
    return pytesseract.image_to_string(img, lang="spa+eng")


def from_image(data: bytes) -> str:
    return _ocr_image(Image.open(io.BytesIO(data)))


def from_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join((p.extract_text() or "") for p in reader.pages[:MAX_PDF_PAGES]).strip()
    if len(text) >= 40:
        return text
    # PDF escaneado: se convierte cada página en imagen y se usa OCR.
    with tempfile.TemporaryDirectory() as d:
        pdf = Path(d) / "in.pdf"
        pdf.write_bytes(data)
        subprocess.run(["pdftoppm", "-r", "200", "-png", "-l", str(MAX_PDF_PAGES), str(pdf), str(Path(d) / "p")],
                       check=True, timeout=60)
        return "\n".join(_ocr_image(Image.open(p)) for p in sorted(Path(d).glob("p*.png")))


def extract(filename: str, content_type: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf") or content_type == "application/pdf":
        return from_pdf(data)
    return from_image(data)
