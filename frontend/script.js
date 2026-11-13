/* ============================================================
   ShieldDefence — interfaz conectada al modelo real
   Basada en el prototipo del Hito 2. Ahora el análisis lo hace el
   backend: la LSTM decide si es phishing y el LLM lo explica.
   El saludo y las respuestas de conversación siguen siendo locales.
   ============================================================ */

const chat = document.getElementById("chat");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const attachBtn = document.getElementById("attach");
const fileInput = document.getElementById("file");
const greeting = document.getElementById("greeting");

let busy = false;

/* ---------- Saludo según la hora ---------- */
(function setGreeting() {
  const h = new Date().getHours();
  const momento = h < 12 ? "Buenos días" : h < 19 ? "Buenas tardes" : "Buenas noches";
  greeting.textContent = `${momento}, pega un correo o sube un pantallazo para revisarlo.`;
})();

/* ---------- Utilidades ---------- */
function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function normalize(text) {
  return text.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");
}

function looksLikeEmail(text) {
  return text.length > 40 || /(https?:\/\/|www\.)/i.test(text);
}

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

/* ---------- Respuesta del modelo ---------- */
function buildAnalysisHtml(r) {
  // Se limita a 1–99 %: un modelo nunca está seguro del todo.
  const pct = (x) => Math.min(99, Math.max(1, Math.round(x * 100)));
  const pctPhish = pct(r.probabilidad_phishing);
  const pctConf = pct(r.confianza);
  const title = r.es_phishing
    ? `Posible phishing detectado — ${r.etiqueta} (${pctPhish}%)`
    : `${r.etiqueta} — confianza del modelo ${pctConf}%`;
  const bar = `<div class="risk-bar"><span class="${r.nivel}" data-width="${Math.max(pctPhish, 4)}"></span></div>`;

  const e = r.explicacion || {};
  const items = (e.senales || []).map((s) => {
    const m = s.match(/^([^:]{3,60}):\s*(.+)$/);
    return m ? `<li><strong>${escapeHtml(m[1])}:</strong> ${escapeHtml(m[2])}</li>` : `<li>${escapeHtml(s)}</li>`;
  }).join("");

  const extracted = r.texto_extraido
    ? `<details><summary>Ver el texto que se leyó del archivo</summary><pre>${escapeHtml(r.texto_extraido)}</pre></details>`
    : "";
  const fuente = r.fuente_explicacion === "llm" ? "explicación generada por el LLM" : "explicación por plantilla";

  return `
    <div class="verdict ${r.nivel}">${escapeHtml(title)}</div>
    ${bar}
    <p>${escapeHtml(e.explicacion || "")}</p>
    ${items ? `<ul>${items}</ul>` : ""}
    <p class="rec-title">Recomendación:</p>
    <p>${escapeHtml(e.recomendacion || "")}</p>
    ${extracted}
    <div class="meta">Clasificación: LSTM · ${fuente} · ${r.tiempos_ms.total} ms</div>`;
}

async function callApi(url, options) {
  const res = await fetch(url, options);
  let data = null;
  try { data = await res.json(); } catch { /* respuesta sin JSON */ }
  if (!res.ok) {
    const detail = data && data.detail ? data.detail : `Error ${res.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

/* ---------- Mensajes en pantalla ---------- */
function addMessage(type, html) {
  const el = document.createElement("div");
  el.className = `msg ${type}`;
  el.innerHTML = `<div class="author">${type === "user" ? "Tú" : "ShieldAI"}</div><div class="body">${html}</div>`;
  messages.appendChild(el);
  scrollToBottom();
  return el;
}

function showTyping() {
  return addMessage("ai", '<div class="typing"><i></i><i></i><i></i></div>');
}

function animateBars(container) {
  container.querySelectorAll(".risk-bar span").forEach((bar) => {
    requestAnimationFrame(() => {
      bar.style.width = bar.dataset.width + "%";
    });
  });
}

function setBusy(v) {
  busy = v;
  attachBtn.disabled = v;
  sendBtn.disabled = v || input.value.trim() === "";
}

async function runAnalysis(userHtml, request) {
  welcome.classList.add("hidden");
  addMessage("user", userHtml);
  const typing = showTyping();
  setBusy(true);
  try {
    const r = await request();
    typing.querySelector(".body").innerHTML = buildAnalysisHtml(r);
    animateBars(typing);
  } catch (err) {
    typing.querySelector(".body").innerHTML =
      `<p class="error">No pude analizar el correo: ${escapeHtml(err.message)}</p>`;
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
    setTimeout(() => {
      typing.querySelector(".body").innerHTML = local;
      scrollToBottom();
    }, 500);
    return;
  }
  runAnalysis(escapeHtml(text), () =>
    callApi("/api/analizar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texto: text }),
    })
  );
}

function handleFile(file) {
  if (!file) return;
  if (file.size > 8 * 1024 * 1024) {
    addMessage("ai", '<p class="error">El archivo pesa más de 8 MB. Prueba con un pantallazo más liviano.</p>');
    return;
  }
  const body = new FormData();
  body.append("archivo", file);
  runAnalysis(`📎 ${escapeHtml(file.name)}`, () => callApi("/api/analizar-archivo", { method: "POST", body }));
}

/* ---------- Eventos ---------- */
function autoResize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}

input.addEventListener("input", () => {
  sendBtn.disabled = busy || input.value.trim() === "";
  autoResize();
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
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
fileInput.addEventListener("change", () => {
  handleFile(fileInput.files[0]);
  fileInput.value = "";
});

/* Pegar una imagen directamente en la caja de texto también funciona. */
input.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item && !busy) {
    e.preventDefault();
    handleFile(item.getAsFile());
  }
});
