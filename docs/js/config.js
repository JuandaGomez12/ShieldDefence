/* De dónde se descargan los modelos. Los dos están en Hugging Face, que permite descargarlos
   desde cualquier página. Para probar con archivos locales: index.html?lstm=model/ */
const params = new URLSearchParams(location.search);

export const LSTM_BASE = params.get("lstm") || "https://huggingface.co/juandaGomez/shielddefence-lstm/resolve/main/";
export const LLM_REPO = params.get("llm_repo") || "juandaGomez/shielddefence-explicador-onnx";
export const LLM_MAX_TOKENS = 220;
/* El LLM en el navegador todavía no da buenos resultados (con WebGPU genera texto sin sentido y en
   CPU es muy lento), así que por ahora queda oculto. Para probarlo: index.html?llm=1 */
export const LLM_ENABLED = params.has("llm");
export const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
export const TESSERACT_URL = "https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.esm.min.js";
export const PDFJS_URL = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.min.mjs";
export const PDFJS_WORKER = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs";
