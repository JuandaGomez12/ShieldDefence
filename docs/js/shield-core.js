/* ============================================================
   ShieldDefence — núcleo en JavaScript
   Versión para el navegador de shield_core/ (Python): limpieza del texto,
   señales, prompt del LLM, plantillas y la BiLSTM. Debe dar los mismos
   resultados que Python (lo comprueba web/tests/parity.mjs).
   ============================================================ */

/* ---------------- preprocess.py ---------------- */
const MAX_CHARS = 6000;
const NOISE = [
  /^\s*\[[^\]]{1,40}\]\s*/gm,
  /_{5,}[\s\S]*?(mailing list|unsubscribe|listinfo)[\s\S]*/gi,
  /-{3,}\s*original message\s*-{3,}[\s\S]*/gi,
  /^(from|to|cc|sent|subject|date)\s*:.*$/gim,
];
const URL_MODEL_RE = /(https?:\/\/[^\s<>"']+|www\.[^\s<>"']+)/gi;
const EMAIL_RE = /[\p{L}\p{N}_.+-]+@[\p{L}\p{N}_-]+\.[\p{L}\p{N}_.-]+/gu;

export function cleanRaw(text) {
  if (typeof text !== "string") return "";
  text = [...text].slice(0, MAX_CHARS * 2).join("");
  for (const rx of NOISE) text = text.replace(rx, " ");
  text = text.replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
  return [...text].slice(0, MAX_CHARS).join("");
}

function domainTokens(url) {
  const low = url.toLowerCase();
  const host = low.replace(/^(https?:\/\/)?(www\.)?/, "").split("/")[0].split("?")[0];
  const toks = ["urltoken"];
  if (low.startsWith("http://")) toks.push("httpplain");
  if (/^\d{1,3}(\.\d{1,3}){3}(:\d+)?$/.test(host)) toks.push("ipurl");
  const label = host ? host.split(".")[0] : "";
  if (/[a-z]\d|\d[a-z]/.test(label.replace(/-/g, ""))) toks.push("mixeddomain");
  if (label.includes("-")) toks.push("hyphendomain");
  const tld = host.includes(".") ? host.slice(host.lastIndexOf(".") + 1) : "";
  if (tld) toks.push("tld" + tld);
  return " " + toks.join(" ") + " ";
}

/** Texto normalizado que recibe la LSTM (igual a for_model en Python). */
export function forModel(text) {
  text = cleanRaw(text).normalize("NFKC").toLowerCase();
  text = text.replace(URL_MODEL_RE, (m) => domainTokens(m));
  text = text.replace(EMAIL_RE, " emailtoken ");
  text = text.replace(/\p{Nd}+/gu, " 0 ");
  text = text.replace(/[^\p{L}\p{N}_\s!?$€%]/gu, " ");
  return text.replace(/\s+/g, " ").trim();
}

/* ---------------- signals.py ---------------- */
const BRANDS = [
  "bancolombia", "davivienda", "nequi", "daviplata", "bbva", "banco de bogota", "paypal", "amazon",
  "netflix", "google", "microsoft", "outlook", "office 365", "apple", "icloud", "facebook",
  "instagram", "whatsapp", "mercadolibre", "mercado libre", "dhl", "fedex", "servientrega",
  "visa", "mastercard", "dian", "chase", "wells fargo", "bank of america", "ebay", "onedrive",
];
const SHORTENERS = new Set(["bit.ly", "tinyurl.com", "t.co", "goo.gl", "cutt.ly", "is.gd", "ow.ly", "rb.gy", "shorturl.at"]);
const ODD_TLDS = [".xyz", ".top", ".click", ".tk", ".ru", ".icu", ".live", ".info", ".work", ".cn", ".zip", ".biz"];
const SECURITY_WORDS = ["seguridad", "verificacion", "verifica", "verify", "secure", "login", "signin", "acceso",
  "soporte", "support", "account", "cuenta", "update", "actualiza", "validar", "validate",
  "confirm", "pagos", "payment", "billing", "banco", "bank", "portal", "nomina"];
const URL_RE = /(https?:\/\/[^\s<>"')]+|www\.[^\s<>"')]+)/gi;

const title = (s) => s.replace(/\b\w/g, (c) => c.toUpperCase());
const norm = (s) => s.toLowerCase().normalize("NFD").replace(/\p{Mn}/gu, "");
const first = (rx, t) => { const m = t.match(rx); return m ? m[0] : ""; };

export function findUrls(text) {
  return [...(text || "").matchAll(URL_RE)].map((m) => m[0].replace(/[.,;:)\]]+$/, ""));
}

function hostOf(url) {
  try {
    return new URL(url.toLowerCase().startsWith("http") ? url : "http://" + url).hostname.toLowerCase();
  } catch {
    return "";
  }
}

const TEXT_RULES = [
  ["urgency", "Lenguaje de urgencia", 20, false,
    /\b(24|48|72) ?(horas|hours|hrs)\b|inmediatamente|de inmediato|urgente|ahora mismo|ultimo aviso|ultima oportunidad|expira|lo antes posible|pague ahora|paga ahora|immediate(ly)?|urgent(ly)?|as soon as possible|within \d+ (hours|days)|final notice|act now|expires? (today|soon)|prompt(ly)? action/,
    (e) => `presiona con prisa ("${e}"), una técnica común para que la persona actúe sin pensar.`],
  ["threat", "Amenaza de suspensión", 20, false,
    /suspendid\w*|suspension|bloquead\w*|bloqueo|cancelad\w*|sera eliminad\w*|acceso restringido|desactivad\w*|suspended|locked|terminated|deactivated|restricted|will be closed|disabled/,
    (e) => `usa el miedo a perder la cuenta o el servicio ("${e}") para presionar.`],
  ["credentials", "Pide datos sensibles", 20, false,
    /(verificar|verifique|confirmar|confirme|actualizar|actualice|ingresa\w*|ingrese) (sus|tus|su|tu) (datos|identidad|cuenta|clave|contrasena|informacion)|contrasena|\bclave\b|\bpin\b|\bcvv\b|numero de (la )?tarjeta|datos bancarios|cuenta bancaria|usuario y contrasena|password|credentials|social security|bank account|card number|verify your (account|identity)|(verify|confirm|update|validate) your (account|identity|details|information|payment|billing|transaction)/,
    (e) => `pide información personal o bancaria ("${e}"), algo que las entidades serias no piden por correo.`],
  ["generic_greeting", "Saludo genérico", 10, false,
    /estimad[oa]\(?a?\)? (cliente|usuario)|querid[oa] (usuario|amigo|cliente)|apreciad[oa] cliente|dear (customer|user|member|friend|sir|beneficiary|account holder)|valued customer/,
    (e) => `no usa el nombre de la persona ("${e}"), algo normal en correos masivos fraudulentos.`],
  ["prize", "Promesa de dinero o premio", 15, false,
    /has ganado|ha ganado|premio|sorteo|herencia|reembolso|devolucion|saldo a favor|regalo gratis|you (have )?won|winner|lottery|prize|inheritance|refund|compensation|claim your/,
    (e) => `ofrece dinero o premios inesperados ("${e}") para llamar la atención.`],
  ["money_request", "Pide mover dinero", 20, false,
    /tarjetas? de regalo|gift ?cards?|transferi\w*|consigna\w*|western union|moneygram|bitcoin|criptomoneda\w*|wire transfer|send (the )?money|bank transfer|ganancias? .{0,20}garantizad\w*|guaranteed (returns?|profit)|million (dollars|usd)|\$ ?\d[\d.,]* (million|millones)/,
    (e) => `busca que la persona envíe dinero, compre algo o invierta ("${e}").`],
  ["shared_doc", "Documento compartido con enlace externo", 15, true,
    /(te|le|les) (comparto|compartio|ha compartido|envio) .{0,25}?(archivo|documento|formato|enlace|factura)|documento compartido|shared (a )?(file|document) with you|has shared|view (the )?document|inicia sesion con tu cuenta|sign in to (view|access)/,
    (e) => `invita a abrir un archivo en un enlace externo ("${e}"), una forma común de robar contraseñas.`],
  ["secrecy", "Pide discreción o evita hablar", 15, false,
    /no (puedo|podre) (hablar|contestar)|estoy en una reunion|que quede entre nosotros|confidencial|keep (this )?(confidential|between us)|i am in a meeting|can'?t talk/,
    (e) => `evita que la persona confirme por otro medio ("${e}"), típico del fraude del jefe.`],
  ["deadline", "Pone una fecha límite", 10, true,
    /antes del? (lunes|martes|miercoles|jueves|viernes|sabado|domingo|\d{1,2})\b|(before|by) (monday|tuesday|wednesday|thursday|friday|tomorrow|end of (the )?day)/,
    (e) => `pide hacer algo antes de una fecha ("${e}") junto con un enlace, para que no lo pienses mucho.`],
  ["click", "Pide hacer clic en un enlace", 5, true,
    /haga clic|haz clic|click (here|below|on the (following )?link|the link)|ingrese al siguiente enlace|ingresa al enlace|siguiente enlace|follow the link|revisalo (aqui|en)/,
    (e) => `insiste en abrir un enlace ("${e}").`],
];

/** Señales encontradas en el correo (igual a detect en Python). */
export function detect(text) {
  text = text || "";
  const t = norm(text);
  const out = [];
  const add = (key, ttl, detail, weight, evidence = "") => {
    if (!out.some((s) => s.key === key)) out.push({ key, title: ttl, detail, weight, evidence });
  };
  const urls = findUrls(text);
  for (const url of urls) {
    const host = hostOf(url);
    if (!host) continue;
    const base = host.startsWith("www.") ? host.slice(4) : host;
    const label = base.split(".")[0];
    if (/[a-z][0-9]|[0-9][a-z]/.test(label.replace(/-/g, ""))) {
      const clean = label.replace(/0/g, "o").replace(/1/g, "l").replace(/3/g, "e").replace(/5/g, "s");
      const brand = BRANDS.find((b) => clean.includes(b.replace(/ /g, "")));
      add("mixed_domain", "Dominio sospechoso",
        `el enlace ${base} cambia letras por números` + (brand ? ` para parecerse a ${title(brand)}` : "") +
        ", algo típico de los sitios falsos.", 30, base);
    }
    const brandInHost = BRANDS.find((b) => base.replace(/-/g, "").includes(b.replace(/ /g, "")));
    const dots = (base.match(/\./g) || []).length;
    const odd = ODD_TLDS.some((x) => base.endsWith(x));
    if (brandInHost && (label.includes("-") || dots >= 2 || odd)) {
      add("brand_lookalike", "Dominio que imita una marca",
        `el enlace ${base} usa el nombre de ${title(brandInHost)}, pero no es su dominio oficial.`, 25, base);
    }
    if (label.includes("-") && SECURITY_WORDS.some((w) => base.includes(w))) {
      add("security_domain", "Dominio con palabras de verificación",
        `el enlace ${base} mezcla guiones con palabras como "seguridad" o "login", algo común en sitios falsos.`, 15, base);
    }
    if (SHORTENERS.has(base)) add("shortener", "Enlace acortado", `el enlace ${base} esconde a qué página lleva realmente.`, 20, base);
    if (odd) add("odd_tld", "Terminación de dominio poco habitual", `la terminación de ${base} se usa mucho en sitios fraudulentos.`, 15, base);
    if (/^\d{1,3}(\.\d{1,3}){3}$/.test(base)) add("ip_url", "Enlace con dirección IP", "el enlace apunta a una dirección IP y no a un dominio oficial.", 25, base);
    if (url.toLowerCase().startsWith("http://")) add("no_https", "Conexión no segura", "el enlace no usa HTTPS, así que la conexión no está cifrada.", 10, base);
  }
  for (const [key, ttl, weight, needsUrl, rx, detail] of TEXT_RULES) {
    const ev = first(rx, t);
    if (ev && (!needsUrl || urls.length)) add(key, ttl, detail(ev), weight, ev);
  }
  return out;
}

/* ---------------- explain.py ---------------- */
export const SYSTEM =
  "Eres ShieldAI, el asistente de ShieldDefence. Recibes un correo, el resultado de un " +
  "clasificador LSTM y las señales encontradas en el texto. Explica en español sencillo, " +
  "sin inventar datos que no estén en el correo, por qué el correo parece phishing o no, y " +
  "da una recomendación práctica. Nunca contradigas la decisión del clasificador. " +
  "Responde con este formato:\nExplicación: ...\nSeñales:\n- ...\nRecomendación: ...";

/** El mensaje que recibe el LLM. Tiene que ser idéntico al del entrenamiento. */
export function buildPrompt(email, isPhishing, prob, signals) {
  signals = signals || detect(email);
  const verdict = isPhishing ? "PHISHING" : "LEGÍTIMO";
  const conf = isPhishing ? prob : 1 - prob;
  const sig = signals.map((s) => `- ${s.title}: ${s.evidence}`).join("\n") || "- (ninguna señal clara por reglas)";
  let body = email.trim();
  const chars = [...body];
  if (chars.length > 1800) body = chars.slice(0, 1800).join("") + " [...]";
  return `Correo:\n<<<\n${body}\n>>>\n\n` +
    `Resultado del clasificador: ${verdict} (confianza ${Math.round(conf * 100)}%)\n` +
    `Señales encontradas por reglas:\n${sig}`;
}

const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
const OPEN_PHISH = [
  "Este correo tiene varias características de un intento de phishing.",
  "El clasificador marcó este correo como phishing y hay detalles en el texto que lo respaldan.",
  "Hay que tener cuidado con este mensaje: tiene la forma típica de un correo de phishing.",
  "Este mensaje parece un intento de engaño para que entregues datos o dinero.",
];
const OPEN_PHISH_NOSIG = [
  "El clasificador marcó este correo como phishing, aunque no tiene las señales más obvias. Los correos hechos con IA suelen verse normales, así que vale la pena revisarlo con calma.",
  "Aunque el correo está bien escrito y no tiene errores, el modelo lo considera sospechoso por su forma y por lo que le pide al lector.",
];
const OPEN_LEGIT = [
  "Este correo no muestra señales claras de phishing.",
  "El clasificador considera que este correo es legítimo.",
  "No encontré nada en este mensaje que indique un intento de engaño.",
];
const REC_PHISH = [
  "No abras los enlaces ni respondas con datos personales. Si tienes dudas, entra tú mismo a la página o aplicación oficial de la entidad, o comunícate por sus canales oficiales.",
  "No hagas clic ni descargues nada. Si crees que puede ser real, verifica directamente con la entidad usando un número o página que ya conozcas, no los que vienen en el correo.",
  "Lo mejor es no responder y reportarlo como phishing en tu correo. Nunca compartas claves, códigos ni datos de tarjetas por este medio.",
];
const REC_MONEY = ["No envíes dinero ni compres tarjetas de regalo. Confirma la solicitud llamando directamente a la persona por un número que ya tengas."];
const REC_LEGIT = [
  "Puedes continuar con normalidad, pero si el correo te pide algo inesperado, confírmalo por otro medio.",
  "Parece seguro, aunque siempre es buena idea revisar el remitente antes de abrir enlaces o adjuntos.",
  "No hace falta hacer nada especial. Aun así, nunca compartas contraseñas o códigos aunque el correo parezca real.",
];

function legitPoints(email, signals) {
  const keys = new Set(signals.map((s) => s.key));
  const pts = [];
  if (!keys.has("credentials") && !keys.has("money_request")) pts.push("No pide contraseñas, códigos, datos bancarios ni dinero.");
  if (!keys.has("urgency") && !keys.has("threat") && !keys.has("deadline")) pts.push("No presiona con plazos cortos ni amenazas.");
  if (!/https?:\/\/|www\./i.test(email)) pts.push("No trae enlaces para hacer clic.");
  if (!pts.length) pts.push("El contenido es coherente con un mensaje normal de su tipo.");
  return pts;
}

/** Explicación por plantilla: se muestra al instante y es el respaldo si el LLM no está. */
export function templateExplanation(email, isPhishing, prob, signals) {
  signals = signals || detect(email);
  const byWeight = [...signals].sort((a, b) => b.weight - a.weight);
  if (isPhishing) {
    let points = byWeight.slice(0, 5).map((s) => `${s.title}: ${s.detail}`);
    if (!points.length) points = ["El estilo y lo que pide el correo se parecen a los correos de phishing con los que se entrenó el modelo."];
    const money = signals.some((s) => s.key === "money_request" || s.key === "secrecy");
    return { explicacion: pick(signals.length ? OPEN_PHISH : OPEN_PHISH_NOSIG), senales: points, recomendacion: pick(money ? REC_MONEY : REC_PHISH) };
  }
  const pts = legitPoints(email, signals).sort(() => Math.random() - 0.5).slice(0, 2);
  for (const s of byWeight.slice(0, 2)) {
    pts.push(`Tiene un detalle a revisar (${s.title.toLowerCase()}: "${s.evidence}"), pero el modelo no lo considera suficiente para decir que es fraude.`);
  }
  return {
    explicacion: signals.length ? "El clasificador considera que este correo es legítimo, aunque tiene algunos detalles que conviene revisar." : pick(OPEN_LEGIT),
    senales: pts,
    recomendacion: pick(REC_LEGIT),
  };
}

/** Convierte la salida del LLM al formato de la interfaz. null si no sirve. */
export function parseAnswer(text) {
  const mExp = text.match(/Explicaci[oó]n:\s*([\s\S]+?)(?=\n\s*Se[ñn]ales:|\n\s*Recomendaci[oó]n:|$(?![\s\S]))/i);
  const mRec = text.match(/Recomendaci[oó]n:\s*([\s\S]+)/i);
  const mSig = text.match(/Se[ñn]ales:\s*([\s\S]+?)(?=\n\s*Recomendaci[oó]n:|$(?![\s\S]))/i);
  if (!mExp || !mRec) return null;
  const sig = (mSig ? mSig[1] : "").split(/\r?\n/).map((l) => l.replace(/^\s*[-*•]\s*/, "").trim()).filter(Boolean);
  return { explicacion: mExp[1].trim(), senales: sig.slice(0, 6), recomendacion: mRec[1].trim().split("\n\n")[0] };
}

/** El LLM no puede contradecir a la LSTM (regla del diseño). */
export function consistent(parsed, isPhishing) {
  const t = (parsed.explicacion + " " + parsed.recomendacion).toLowerCase();
  if (isPhishing) return !/\b(es|parece) (un correo )?leg[ií]timo\b|no (es|parece) phishing|no hay se[ñn]ales/.test(t);
  return !/\b(es|parece) (un )?(intento de )?phishing\b|es (un )?fraude|es una estafa/.test(t);
}

/* ---------------- lstm_numpy.py ---------------- */
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * Math.pow(2, -14) * (f / 1024)
      : e === 31 ? (f ? NaN : s * Infinity)
        : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

/** La BiLSTM entrenada con Keras: Embedding -> BiLSTM -> Dense -> Dense sigmoide. */
export class BiLSTM {
  /** meta: lstm_web.json, buffer: ArrayBuffer de lstm_web.bin */
  constructor(meta, buffer) {
    this.maxLen = meta.max_len;
    this.threshold = meta.umbral;
    this.vocab = new Map(meta.vocab.map((t, i) => [t, i]));
    this.t = {};
    for (const [name, spec] of Object.entries(meta.tensors)) {
      const arr = spec.dtype === "float16"
        ? f16ToF32(new Uint16Array(buffer, spec.offset, spec.size))
        : new Float32Array(buffer, spec.offset, spec.size);
      this.t[name] = { data: arr, shape: spec.shape };
    }
    this.units = this.t.fw_recurrent.shape[0];
    this.emb = this.t.embedding.shape[1];
  }

  tokens(modelText) {
    return modelText.split(/\s+/).filter(Boolean).slice(0, this.maxLen)
      .map((w) => (this.vocab.has(w) ? this.vocab.get(w) : 1)).filter((i) => i !== 0);
  }

  run(ids, prefix) {
    const U = this.units, E = this.emb, G = 4 * U;
    const K = this.t[prefix + "_kernel"].data, R = this.t[prefix + "_recurrent"].data, B = this.t[prefix + "_bias"].data;
    const emb = this.t.embedding.data;
    let h = new Float32Array(U), c = new Float32Array(U);
    const z = new Float32Array(G);
    for (const id of ids) {
      const x = id * E;
      for (let g = 0; g < G; g++) z[g] = B[g];
      for (let e = 0; e < E; e++) {
        const xv = emb[x + e];
        if (xv === 0) continue;
        const row = e * G;
        for (let g = 0; g < G; g++) z[g] += xv * K[row + g];
      }
      for (let u = 0; u < U; u++) {
        const hv = h[u];
        if (hv === 0) continue;
        const row = u * G;
        for (let g = 0; g < G; g++) z[g] += hv * R[row + g];
      }
      const nh = new Float32Array(U);
      for (let u = 0; u < U; u++) {
        const i = sigmoid(z[u]), f = sigmoid(z[U + u]), gg = Math.tanh(z[2 * U + u]), o = sigmoid(z[3 * U + u]);
        c[u] = f * c[u] + i * gg;
        nh[u] = o * Math.tanh(c[u]);
      }
      h = nh;
    }
    return h;
  }

  predictIds(ids) {
    if (!ids.length) ids = [1];
    const hf = this.run(ids, "fw"), hb = this.run([...ids].reverse(), "bw");
    const h = new Float32Array(hf.length * 2);
    h.set(hf); h.set(hb, hf.length);
    const DK = this.t.dense_kernel.data, DB = this.t.dense_bias.data, D = DB.length;
    const d = new Float32Array(D);
    for (let j = 0; j < D; j++) {
      let s = DB[j];
      for (let i = 0; i < h.length; i++) s += h[i] * DK[i * D + j];
      d[j] = Math.max(0, s);
    }
    const OK = this.t.out_kernel.data;
    let s = this.t.out_bias.data[0];
    for (let j = 0; j < D; j++) s += d[j] * OK[j];
    return sigmoid(s);
  }

  /** Probabilidad de que el correo sea phishing. */
  predict(rawEmail) {
    return this.predictIds(this.tokens(forModel(rawEmail)));
  }
}
