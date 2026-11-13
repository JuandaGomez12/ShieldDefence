/* Lectura de pantallazos (OCR con Tesseract.js) y PDF (PDF.js), todo en el navegador. */
import { TESSERACT_URL, PDFJS_URL, PDFJS_WORKER } from "./config.js";

let ocrWorker = null;

async function ocr(image, onStatus) {
  if (!ocrWorker) {
    onStatus("Cargando el lector de imágenes (solo la primera vez)...");
    const { createWorker } = await import(TESSERACT_URL);
    ocrWorker = await createWorker(["spa", "eng"]);
  }
  onStatus("Leyendo el texto de la imagen...");
  const { data } = await ocrWorker.recognize(image);
  return data.text;
}

/* Las imágenes pequeñas se agrandan: Tesseract lee mejor con letras grandes. */
async function prepareImage(blob) {
  const bmp = await createImageBitmap(blob);
  const scale = bmp.width < 1400 ? Math.min(3, 1400 / bmp.width) : 1;
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bmp.width * scale);
  canvas.height = Math.round(bmp.height * scale);
  canvas.getContext("2d").drawImage(bmp, 0, 0, canvas.width, canvas.height);
  return canvas;
}

async function readPdf(file, onStatus) {
  onStatus("Abriendo el PDF...");
  const pdfjs = await import(PDFJS_URL);
  pdfjs.GlobalWorkerOptions.workerSrc = PDFJS_WORKER;
  const doc = await pdfjs.getDocument({ data: await file.arrayBuffer() }).promise;
  const pages = Math.min(doc.numPages, 5);
  let text = "";
  for (let i = 1; i <= pages; i++) {
    const content = await (await doc.getPage(i)).getTextContent();
    text += content.items.map((it) => it.str + (it.hasEOL ? "\n" : " ")).join("") + "\n";
  }
  if (text.replace(/\s/g, "").length >= 40) return text;
  // PDF escaneado: se dibujan las páginas y se les hace OCR.
  text = "";
  for (let i = 1; i <= Math.min(pages, 3); i++) {
    const page = await doc.getPage(i);
    const vp = page.getViewport({ scale: 2 });
    const canvas = document.createElement("canvas");
    canvas.width = vp.width; canvas.height = vp.height;
    await page.render({ canvasContext: canvas.getContext("2d"), viewport: vp }).promise;
    text += (await ocr(canvas, onStatus)) + "\n";
  }
  return text;
}

export async function extractText(file, onStatus = () => {}) {
  if (file.type === "application/pdf" || /\.pdf$/i.test(file.name)) return readPdf(file, onStatus);
  if (file.type.startsWith("image/")) return ocr(await prepareImage(file), onStatus);
  throw new Error("Solo se aceptan imágenes (PNG, JPG, WEBP) o PDF.");
}
