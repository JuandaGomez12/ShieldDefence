"""Limpieza de texto compartida entre el entrenamiento y la API.

La misma función se usa al entrenar la LSTM y al analizar un correo nuevo,
así el modelo siempre ve el texto preparado de la misma forma.
"""
import re
import unicodedata

URL_RE = re.compile(r"(https?://[^\s<>\"']+|www\.[^\s<>\"']+)", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
NUM_RE = re.compile(r"\d+")
MAX_CHARS = 6000

# Restos de las listas de correo y de los datasets que no dicen nada sobre
# si un correo es phishing, pero que el modelo podría memorizar.
_NOISE = [
    re.compile(r"^\s*\[[^\]]{1,40}\]\s*", re.M),                     # [lista] en el asunto
    re.compile(r"_{5,}.*?(mailing list|unsubscribe|listinfo).*", re.I | re.S),
    re.compile(r"-{3,}\s*original message\s*-{3,}.*", re.I | re.S),
    re.compile(r"^(from|to|cc|sent|subject|date)\s*:.*$", re.I | re.M),
]


def clean_raw(text: str) -> str:
    """Quita ruido de formato pero deja el correo legible (sirve para el LLM)."""
    if not isinstance(text, str):
        return ""
    text = text[:MAX_CHARS * 2]
    for rx in _NOISE:
        text = rx.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:MAX_CHARS]


def _domain_tokens(url: str) -> str:
    """Convierte un enlace en fichas útiles: 'urltoken' y señales del dominio."""
    host = re.sub(r"^(https?://)?(www\.)?", "", url.lower()).split("/")[0].split("?")[0]
    toks = ["urltoken"]
    if url.lower().startswith("http://"):
        toks.append("httpplain")
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}(:\d+)?", host):
        toks.append("ipurl")
    label = host.split(".")[0] if host else ""
    if re.search(r"[a-z]\d|\d[a-z]", label.replace("-", "")):
        toks.append("mixeddomain")
    if "-" in label:
        toks.append("hyphendomain")
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    if tld:
        toks.append("tld" + tld)
    return " " + " ".join(toks) + " "


def for_model(text: str) -> str:
    """Texto normalizado que recibe la LSTM."""
    text = clean_raw(text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = URL_RE.sub(lambda m: _domain_tokens(m.group(0)), text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = NUM_RE.sub(" 0 ", text)
    text = re.sub(r"[^\w\sáéíóúüñ!?$€%]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
