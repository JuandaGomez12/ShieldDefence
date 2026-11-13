/* ============================================================
   ShieldDefence — versión que corre completa en el navegador
   La BiLSTM decide si el correo es phishing (shield-core.js), las reglas
   buscan señales y el LLM ajustado con LoRA explica el resultado (llm-worker.js).
   Ningún correo sale del computador del usuario.
   ============================================================ */
import {
  cleanRaw, detect, BiLSTM, SYSTEM, buildPrompt, templateExplanation, parseAnswer, consistent,
} from "./shield-core.js";
import { LSTM_BASE, LLM_REPO, LLM_MAX_TOKENS, TRANSFORMERS_URL } from "./config.js";
import { extractText } from "./files.js";

const $ = (id) => document.getElementById(id);
const chat = $("chat"), messages = $("messages"), welcome = $("welcome"), form = $("form");
const input = $("input"), sendBtn = $("send"), attachBtn = $("attach"), fileInput = $("file");
const statusEl = $("status"), statusText = $("status-text"), llmBtn = $("llm-toggle");

let busy = false;
let lstm = null;

/* ---------- Saludo según la hora ---------- */
(function setGreeting() {
  const h = new Date().getHours();
  const momento = h < 12 ? "Buenos días" : h < 19 ? "Buenas tardes" : "Buenas noches";
  $("greeting").textContent = `${momento}, pega un correo o sube un pantallazo para revisarlo.`;
})();

function setStatus(state, text) {
  statusEl.className = "status " + state;
  statusText.textContent = text;
}

/* ---------- BiLSTM ---------- */
const lstmReady = (async () => {
  setStatus("loading", "Cargando modelo...");
  const [meta, bin] = await Promise.all([
    fetch(LSTM_BASE + "lstm_web.json").then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }),
    fetch(LSTM_BASE + "lstm_web.bin").then((r) => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); }),
  ]);
  lstm = new BiLSTM(meta, bin);
  setStatus("", "En línea");
})().catch((err) => {
  setStatus("error", "Sin modelo");
  console.error(err);
  throw err;
});

/* ---------- LLM (opcional) ---------- */
const llm = {
  worker: null, state: "off", device: "", pending: new Map(), nextId: 1,
  start() {
    if (this.worker) return;
    this.state = "loading";
    updateLlmButton("Descargando el LLM... 0%");
    this.worker = new Worker(new URL("./llm-worker.js", import.meta.url), { type: "module" });
    this.worker.onmessage = (e) => this.onMessage(e.data);
    this.worker.postMessage({ type: "load", repo: LLM_REPO, transformersUrl: TRANSFORMERS_URL });
  },
  onMessage(m) {
    if (m.type === "progress") {
      updateLlmButton(`Descargando el LLM... ${Math.floor((m.loaded / m.total) * 100)}% de ${Math.round(m.total / 2 ** 20)} MB`);
    } else if (m.type === "ready") {
      this.state = "ready"; this.device = m.device;
      updateLlmButton(`LLM activo (${m.device === "webgpu" ? "GPU" : "CPU"})`);
      for (const fn of readyListeners.splice(0)) fn();
    } else if (m.type === "error" && !m.id) {
      this.state = "off"; this.worker.terminate(); this.worker = null;
      updateLlmButton("No se pudo cargar el LLM. Reintentar");
      console.error(m.message);
    } else {
      const p = this.pending.get(m.id);
      if (!p) return;
      if (m.type === "token") p.onToken(m.text);
      else if (m.type === "done") { this.pending.delete(m.id); p.resolve(m); }
      else if (m.type === "error") { this.pending.delete(m.id); p.reject(new Error(m.message)); }
    }
  },
  generate(prompt, onToken) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onToken });
      this.worker.postMessage({ type: "generate", id, prompt, system: SYSTEM, maxTokens: LLM_MAX_TOKENS });
    });
  },
};
const readyListeners = [];

function updateLlmButton(text) {
  llmBtn.textContent = text;
  llmBtn.classList.toggle("on", llm.state === "ready");
  llmBtn.disabled = llm.state === "loading";
}
llmBtn.addEventListener("click", () => {
  if (llm.state === "off") llm.start();
});

/* ---------- Análisis (lo mismo que ShieldDefence.analyze en Python) ---------- */
async function analyze(text) {
  await lstmReady;
  const t0 = performance.now();
  const email = cleanRaw(text);
  const prob = lstm.predict(email);
  const isPhishing = prob >= lstm.threshold;
  const signals = detect(email);
  let nivel = "safe", etiqueta = "Sin señales claras de riesgo";
  if (isPhishing) [nivel, etiqueta] = prob >= 0.85 ? ["high", "Riesgo alto"] : ["mid", "Riesgo medio"];
  return {
    email, signals,
    es_phishing: isPhishing,
    probabilidad_phishing: prob,
    confianza: isPhishing ? prob : 1 - prob,
    nivel, etiqueta,
    explicacion: templateExplanation(email, isPhishing, prob, signals),
    fuente_explicacion: "plantilla",
    ms: Math.round(performance.now() - t0),
  };
}

/* ---------- Utilidades ---------- */
function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const scrollToBottom = () => { chat.scrollTop = chat.scrollHeight; };
const normalize = (t) => t.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
const looksLikeEmail = (t) => t.length > 40 || /(https?:\/\/|www\.)/i.test(t);

/* ---------- Respuestas de conversación (no necesitan el modelo) ---------- */
function localReply(text) {
  const t = normalize(text).trim();
  if (/^(hola|buenas|hey|buenos dias|buenas tardes|buenas noches)\b/.test(t) && text.length < 30) {
    return "<p>¡Hola! Soy ShieldAI. Pega aquí un correo o sube un pantallazo o PDF con el botón del clip y te diré si parece phishing.</p>";
  }
  if (/(gracias|thanks|genial|perfecto)/.test(t) && text.length < 60) {
    return "<p>¡Con gusto! Si recibes otro correo sospechoso, pégalo aquí y lo reviso. 🛡️</p>";
  }
  if (/(que es (el )?phishing|como (me )?protejo|consejos)/.test(t)) {
    return `
      <p>El <strong>phishing</strong> es un engaño en el que alguien se hace pasar por una entidad de confianza para robar tus datos. Consejos básicos:</p>
      <ul>
        <li>Revisa bien el dominio del enlace antes de hacer clic.</li>
        <li>Desconfía de mensajes que te presionan con plazos o amenazas.</li>
        <li>Nunca compartas contraseñas, PIN o códigos por correo o chat.</li>
        <li>Entra siempre a tus servicios desde la app o web oficial.</li>
      </ul>`;
  }
  if (!looksLikeEmail(text)) {
    return "<p>Para ayudarte mejor, pega el contenido completo del correo (asunto y mensaje) o sube un pantallazo.</p>";
  }
  return null;
}

/* ---------- Respuesta en pantalla ---------- */
function explanationHtml(e) {
  const items = (e.senales || []).map((s) => {
    const m = s.match(/^([^:]{3,60}):\s*(.+)$/);
    return m ? `<li><strong>${escapeHtml(m[1])}:</strong> ${escapeHtml(m[2])}</li>` : `<li>${escapeHtml(s)}</li>`;
  }).join("");
  return `
    <p>${escapeHtml(e.explicacion || "")}</p>
    ${items ? `<ul>${items}</ul>` : ""}
    <p class="rec-title">Recomendación:</p>
    <p>${escapeHtml(e.recomendacion || "")}</p>`;
}

function metaText(r) {
  const fuente = r.fuente_explicacion === "llm" ? `explicación generada por el LLM (${(r.llm_ms / 1000).toFixed(1)} s)` : "explicación por plantilla";
  return `Clasificación: LSTM (${r.ms} ms) · ${fuente} · todo se procesó en tu navegador`;
}

function buildAnalysisHtml(r, extracted) {
  const pct = (x) => Math.min(99, Math.max(1, Math.round(x * 100)));  // un modelo nunca está seguro del todo
  const pctPhish = pct(r.probabilidad_phishing);
  const title = r.es_phishing
    ? `Posible phishing detectado — ${r.etiqueta} (${pctPhish}%)`
    : `${r.etiqueta} — confianza del modelo ${pct(r.confianza)}%`;
  const ext = extracted
    ? `<details><summary>Ver el texto que se leyó del archivo</summary><pre>${escapeHtml(extracted)}</pre></details>` : "";
  return `
    <div class="verdict ${r.nivel}">${escapeHtml(title)}</div>
    <div class="risk-bar"><span class="${r.nivel}" data-width="${Math.max(pctPhish, 4)}"></span></div>
    <div class="explanation">${explanationHtml(r.explicacion)}</div>
    ${ext}
    <div class="meta">${metaText(r)}</div>`;
}

/* Cuando el LLM está activo, su explicación reemplaza a la plantilla mientras se escribe.
   Si el formato no sirve o contradice a la LSTM, se deja la plantilla. */
async function explainWithLlm(r, box) {
  const exp = box.querySelector(".explanation"), meta = box.querySelector(".meta");
  meta.textContent = "El LLM está escribiendo la explicación...";
  const prompt = buildPrompt(r.email, r.es_phishing, r.probabilidad_phishing, r.signals);
  try {
    const out = await llm.generate(prompt, (text) => {
      const partial = parseAnswer(text + "\nRecomendación: ...");
      if (partial) { exp.innerHTML = explanationHtml({ ...partial, recomendacion: /Recomendaci[oó]n:/i.test(text) ? parseAnswer(text)?.recomendacion : "..." }); scrollToBottom(); }
    });
    const parsed = parseAnswer(out.text);
    if (parsed && consistent(parsed, r.es_phishing)) {
      r.explicacion = parsed; r.fuente_explicacion = "llm"; r.llm_ms = out.ms;
    } else {
      console.warn("Salida del LLM descartada:", out.text);
    }
  } catch (err) {
    console.error(err);
  }
  exp.innerHTML = explanationHtml(r.explicacion);
  meta.textContent = metaText(r);
}

function addMessage(type, html) {
  const el = document.createElement("div");
  el.className = `msg ${type}`;
  el.innerHTML = `<div class="author">${type === "user" ? "Tú" : "ShieldAI"}</div><div class="body">${html}</div>`;
  messages.appendChild(el);
  scrollToBottom();
  return el;
}
const showTyping = () => addMessage("ai", '<div class="typing"><i></i><i></i><i></i></div>');

function setBusy(v) {
  busy = v;
  attachBtn.disabled = v;
  sendBtn.disabled = v || input.value.trim() === "";
}

async function runAnalysis(userHtml, getText) {
  welcome.classList.add("hidden");
  addMessage("user", userHtml);
  const box = showTyping();
  const body = box.querySelector(".body");
  setBusy(true);
  try {
    const { text, extracted } = await getText((s) => { body.innerHTML = `<p class="meta">${escapeHtml(s)}</p>`; });
    if (cleanRaw(text).length < 20) throw new Error("no encontré suficiente texto para analizar.");
    const r = await analyze(text);
    body.innerHTML = buildAnalysisHtml(r, extracted);
    requestAnimationFrame(() => body.querySelectorAll(".risk-bar span").forEach((b) => { b.style.width = b.dataset.width + "%"; }));
    if (llm.state === "ready") await explainWithLlm(r, body);
    else if (llm.state === "loading") readyListeners.push(() => explainWithLlm(r, body));
  } catch (err) {
    body.innerHTML = `<p class="error">No pude analizar el correo: ${escapeHtml(err.message)}</p>`;
  } finally {
    setBusy(false);
    scrollToBottom();
  }
}

function handleSend(text) {
  const local = localReply(text);
  if (local) {
    welcome.classList.add("hidden");
    addMessage("user", escapeHtml(text));
    const typing = showTyping();
    setTimeout(() => { typing.querySelector(".body").innerHTML = local; scrollToBottom(); }, 500);
    return;
  }
  runAnalysis(escapeHtml(text), async () => ({ text }));
}

function handleFile(file) {
  if (!file) return;
  if (file.size > 8 * 1024 * 1024) {
    addMessage("ai", '<p class="error">El archivo pesa más de 8 MB. Prueba con un pantallazo más liviano.</p>');
    return;
  }
  runAnalysis(`📎 ${escapeHtml(file.name || "imagen pegada")}`, async (status) => {
    const text = await extractText(file, status);
    return { text, extracted: text.trim() };
  });
}

/* ---------- Eventos ---------- */
function autoResize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}
input.addEventListener("input", () => { sendBtn.disabled = busy || input.value.trim() === ""; autoResize(); });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || busy) return;
  input.value = "";
  sendBtn.disabled = true;
  autoResize();
  handleSend(text);
});
attachBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { handleFile(fileInput.files[0]); fileInput.value = ""; });
input.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item && !busy) { e.preventDefault(); handleFile(item.getAsFile()); }
});
