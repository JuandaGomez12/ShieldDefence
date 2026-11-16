"""Limpieza de texto compartida entre el entrenamiento y la API.

La misma función se usa al entrenar la LSTM y al analizar un correo nuevo,
así el modelo siempre ve el texto preparado de la misma forma.
"""
import re
import unicodedata

from .deobfuscate import deobfuscate

URL_RE = re.compile(r"(https?://[^\s<>\"']+|www\.[^\s<>\"']+)", re.I)
# El (?<!...) evita que la búsqueda empiece en medio de una palabra larga (sin eso, una cadena de miles
# de letras sin "@" hacía que la búsqueda fuera lentísima).
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+\.[\w.-]+")
NUM_RE = re.compile(r"\d+")
MAX_CHARS = 6000

# Restos de las listas de correo y de los datasets que no dicen nada sobre
# si un correo es phishing, pero que el modelo podría memorizar.
_LIST_TAG = re.compile(r"^[ \t]*\[[^\]\n]{1,40}\][ \t]*", re.M)        # [lista] en el asunto
_HEADERS = re.compile(r"^(from|to|cc|sent|subject|date)[ \t]*:.*$", re.I | re.M)
_UNDERSCORES = re.compile(r"_{5}")
_LIST_FOOTER = re.compile(r"mailing list|unsubscribe|listinfo", re.I)
_ORIGINAL = re.compile(r"-{3}\s*original message\s*-{3}", re.I)


def _cut_footers(text):
    """Corta el pie de las listas de correo ("_____ ... unsubscribe") y los mensajes reenviados
    ("--- Original Message ---"), recorriendo el texto una sola vez.

    Antes eran expresiones del tipo "_{5,}.*?unsubscribe.*" que, con un correo con miles de guiones
    bajos o guiones seguidos, tardaban minutos (y en el navegador congelaban la página).
    """
    m = _UNDERSCORES.search(text)
    if m:
        last = None
        for last in _LIST_FOOTER.finditer(text):
            pass
        if last is not None and last.start() >= m.end():
            text = text[:m.start()] + " "
    m = _ORIGINAL.search(text)
    if m:
        start = m.start()
        while start > 0 and text[start - 1] == "-":
            start -= 1
        text = text[:start] + " "
    return text


def clean_raw(text: str) -> str:
    """Quita ruido de formato pero deja el correo legible (sirve para el LLM)."""
    if not isinstance(text, str):
        return ""
    text = text[:MAX_CHARS * 2]
    text = _LIST_TAG.sub(" ", text)
    text = _cut_footers(text)
    text = _HEADERS.sub(" ", text)
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
    text = deobfuscate(clean_raw(text))[0]  # "c l a v e", "cl4ve"... -> "clave"
    text = unicodedata.normalize("NFKC", text).lower()
    text = URL_RE.sub(lambda m: _domain_tokens(m.group(0)), text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = NUM_RE.sub(" 0 ", text)
    text = re.sub(r"[^\w\sáéíóúüñ!?$€%]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
