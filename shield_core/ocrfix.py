"""Arregla los enlaces que el OCR lee mal en los pantallazos.

Tesseract suele leer el punto de un dominio como un espacio ("bancolombia-soporte top") y las barras de
"https://" como "https:/I" (o se come los dos puntos: "https //"). Con el enlace roto, las reglas de dominios no se activan. Esto solo toca las
líneas que tienen un enlace o un correo, para no cambiar el resto del texto.
"""
import re

TLDS = ("com|co|net|org|top|xyz|info|live|click|site|app|edu|gov|io|me|link|online|biz|ru|cn|tk|icu|work|zip|"
        "store|shop|club|web|es|mx|ar|pe|cl|ec|us|uk")
SCHEME_RE = re.compile(r"\b(https?)\s*:\s*/\s*[/Il1|]\s*", re.I)
# Sin los dos puntos, o con ";" "." "," en su lugar ("https //sitio"). Aquí el "1" no cuenta como barra,
# para no tocar cosas como "HTTP/1.1".
SCHEME_NOCOLON_RE = re.compile(r"\b(https?)\s*[;.,]?\s*/\s*[/Il|]\s*", re.I)
WWW_RE = re.compile(r"\bwww\s*[.,]\s*", re.I)
TLD_RE = re.compile(r"([A-Za-z0-9-]{2,})\s+(" + TLDS + r")(?=[\s/:.,;)>\]]|$)")
URL_LINE_RE = re.compile(r"https?\s*:|https?\s*[;.,]?\s*/|www\s*[.,]|@", re.I)


def fix_ocr_links(text):
    out = []
    for line in (text or "").split("\n"):
        if URL_LINE_RE.search(line):
            line = SCHEME_RE.sub(lambda m: m.group(1).lower() + "://", line)
            line = SCHEME_NOCOLON_RE.sub(lambda m: m.group(1).lower() + "://", line)
            line = WWW_RE.sub("www.", line)
            for _ in range(2):  # "sitio com co" -> "sitio.com.co"
                line = TLD_RE.sub(r"\1.\2", line)
        out.append(line)
    return "\n".join(out)
