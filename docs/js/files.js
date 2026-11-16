/* Lectura de pantallazos (OCR con Tesseract.js) y PDF (PDF.js), todo en el navegador. */
import { TESSERACT_URL, PDFJS_URL, PDFJS_WORKER } from "./config.js";
import { fixOcrLinks } from "./shield-core.js";

let ocrWorker = null;
let ocrStatus = () => {};

async function ocr(image, onStatus) {
  ocrStatus = onStatus;
  if (!ocrWorker) {
    onStatus("Cargando el lector de imágenes (solo la primera vez)...");
    const mod = await import(TESSERACT_URL);
    const createWorker = mod.createWorker || mod.default.createWorker;
    ocrWorker = await createWorker(["spa", "eng"], 1, {
      logger: (m) => {
        if (m.status === "recognizing text") ocrStatus(`Leyendo el texto de la imagen... ${Math.round(m.progress * 100)}%`);
      },
    });
  }
  onStatus("Leyendo el texto de la imagen...");
  const { data } = await ocrWorker.recognize(image);
  return fixOcrLinks(data.text);  // el OCR lee "sitio.top" como "sitio top" y "https://" como "https:/I"
}

/* Las imágenes pequeñas se agrandan: Tesseract lee mejor con letras grandes. Los pantallazos en modo
   oscuro (letra clara sobre fondo oscuro) se invierten: Tesseract lee peor así y se come signos como ":". */
async function prepareImage(blob) {
  const bmp = await createImageBitmap(blob);
  const scale = bmp.width < 1400 ? Math.min(3, 1400 / bmp.width) : 1;
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bmp.width * scale);
  canvas.height = Math.round(bmp.height * scale);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const px = img.data;
  let sum = 0, n = 0;
  for (let i = 0; i < px.length; i += 4 * 16) { sum += 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]; n++; }
  if (n && sum / n < 110) {
    for (let i = 0; i < px.length; i += 4) { px[i] = 255 - px[i]; px[i + 1] = 255 - px[i + 1]; px[i + 2] = 255 - px[i + 2]; }
    ctx.putImageData(img, 0, 0);
  }
  return canvas;
}

/* Los enlaces de los botones ("Ver documento") no se ven en el texto, pero el PDF los guarda como anotaciones. */
async function pdfLinks(doc, pages) {
  const urls = [];
  for (let i = 1; i <= pages; i++) {
    for (const a of await (await doc.getPage(i)).getAnnotations()) {
      if (a.subtype === "Link" && a.url && !urls.includes(a.url)) urls.push(a.url);
    }
  }
  return urls;
}

export function withHiddenLinks(text, urls) {
  const hidden = urls.filter((u) => !text.includes(u));
  return hidden.length ? `${text.trim()}\n\nEnlaces de los botones del correo:\n${hidden.join("\n")}` : text;
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
  const links = await pdfLinks(doc, pages);
  if (text.replace(/\s/g, "").length >= 40) return withHiddenLinks(text, links);
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
  return withHiddenLinks(text, links);
}

/* Al pegar un correo copiado de Gmail u Outlook, el portapapeles trae el HTML con los enlaces reales de los
   botones. Se agregan al texto para que las reglas revisen a dónde llevan. */
export function linksFromHtml(html) {
  const doc = new DOMParser().parseFromString(html, "text/html");
  const urls = [];
  for (const a of doc.querySelectorAll("a[href]")) {
    const href = a.getAttribute("href").trim();
    if (/^https?:\/\//i.test(href) && !urls.includes(href)) urls.push(href);
  }
  return urls;
}

export async function extractText(file, onStatus = () => {}) {
  if (file.type === "application/pdf" || /\.pdf$/i.test(file.name)) return readPdf(file, onStatus);
  if (file.type.startsWith("image/")) return ocr(await prepareImage(file), onStatus);
  throw new Error("Solo se aceptan imágenes (PNG, JPG, WEBP) o PDF.");
}
