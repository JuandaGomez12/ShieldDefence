"""Quita los disfraces que usan los atacantes para que los filtros no lean las palabras clave.

Ejemplos que se deshacen:
  - letras separadas:        "c l a v e", "c.o.n.t.r.a.s.e.ñ.a", "B-L-O-Q-U-E-A-D-A"
  - números por letras:      "cl4ve", "c0ntraseña", "b4nc0lombia"
  - caracteres invisibles:   "cla​ve" (espacio de ancho cero en medio de la palabra)
  - letras de otro alfabeto: "сlаve" con la "c" y la "a" cirílicas, que se ven iguales
  - letras "de adorno":      "ｃｌａｖｅ" (ancho completo) o "𝐜𝐥𝐚𝐯𝐞" (negrita matemática)

Las mayúsculas no hacen falta: todo se pasa a minúsculas antes.

La función devuelve el texto limpio y qué trucos encontró. El texto limpio es el que ven la BiLSTM y
las reglas; los trucos se vuelven la señal "Texto disfrazado", porque un correo legítimo no escribe así.
Los enlaces no se tocan (salvo quitar invisibles y letras de otro alfabeto), para que las reglas de
dominios sigan viendo "b4nc0lombia" tal cual.
"""
import re
import unicodedata

URL_RE = re.compile(r"(https?://[^\s<>\"']+|www\.[^\s<>\"']+)", re.I)

INVISIBLE_RE = re.compile("[­͏؜ᅟᅠ឴឵᠎​-‏‪-‮"
                          "⁠-⁤⁦-⁯﻿]")
# Letras cirílicas y griegas que se ven iguales a letras latinas.
HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "ԁ": "d", "һ": "h", "ӏ": "l", "ԛ": "q", "ԝ": "w", "ɡ": "g",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "Ү": "Y", "І": "I", "Ј": "J", "Ѕ": "S",
    "ο": "o", "α": "a", "ν": "v", "ι": "i", "κ": "k", "ρ": "p", "τ": "t", "υ": "u", "ε": "e",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Χ": "X", "Υ": "Y",
}
HOMO_RE = re.compile("[" + "".join(HOMOGLYPHS) + "]")
FANCY_RE = re.compile("[！-～\U0001d400-\U0001d7ff]")  # ancho completo y letras matemáticas
# Cuatro o más letras sueltas separadas por espacios, o cinco o más separadas por . - _ * ·
# (con menos se confundiría con siglas como "S.A.S." o palabras sueltas como "y", "a", "o").
SPACED_RE = re.compile(r"(?<![^\W\d_])(?:[^\W\d_] ){3,}[^\W\d_](?![^\W\d_])")
DOTTED_RE = re.compile(r"(?<![^\W\d_])(?:[^\W\d_][.\-_*·]){4,}[^\W\d_](?![^\W\d_])")
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
LEET_TOKEN_RE = re.compile(r"(?<![\w@$])[\w@$]*[0134578@$][\w@$]*(?![\w@$])")
# Solo se corrigen los números por letras cuando el resultado es una palabra típica de phishing.
# Así no se dañan cosas normales como "mp3", "4g", "covid19" o códigos de factura.
WATCH = ("clave", "contras", "passw", "cuent", "banc", "nequi", "davi", "bbva", "paypal", "netfl", "amazon",
         "apple", "micros", "outlook", "googl", "verif", "tarjet", "credit", "urgen", "gratis", "premi", "ganad",
         "ganas", "bloque", "suspend", "segur", "acces", "usuari", "login", "transf", "pago", "factur", "diner",
         "bitcoin", "codig", "datos", "confirm", "regalo", "clic", "enlace", "inmediat", "cvv", "pin")

TRICK_NAMES = {
    "invisibles": "caracteres invisibles dentro de las palabras",
    "otro_alfabeto": "letras de otro alfabeto que se ven iguales a las normales",
    "letras_especiales": "letras de adorno (ancho completo o negrita matemática)",
    "letras_separadas": "letras separadas por espacios o puntos",
    "numeros_por_letras": "números en lugar de letras",
}


def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _watched(word):
    w = _strip_accents(word.lower())
    return len(w) >= 3 and any(w.startswith(k) or (len(k) >= 5 and k in w) for k in WATCH)


def _fix_homoglyphs(word, tricks, examples):
    if HOMO_RE.search(word) and re.search(r"[A-Za-z]", word):
        examples.setdefault("otro_alfabeto", word)
        tricks.add("otro_alfabeto")
        return HOMO_RE.sub(lambda m: HOMOGLYPHS[m.group()], word)
    return word


def _fix_text(part, tricks, examples):
    def spaced(m):
        joined = re.sub(r"[ .\-_*·]", "", m.group())
        tricks.add("letras_separadas")
        examples.setdefault("letras_separadas", m.group())
        return joined

    part = SPACED_RE.sub(spaced, part)
    part = DOTTED_RE.sub(spaced, part)

    def leet(m):
        tok = m.group()
        letters = sum(c.isalpha() for c in tok)
        if letters < 2 or tok.isdigit():
            return tok
        for cand in (tok.translate(LEET), tok.replace("1", "l").translate(LEET)):
            if _watched(cand):
                tricks.add("numeros_por_letras")
                examples.setdefault("numeros_por_letras", tok)
                return cand
        return tok

    return LEET_TOKEN_RE.sub(leet, part)


DEFANG_SCHEME_RE = re.compile(r"\bhxxp(s?)(://)", re.I)
DEFANG_DOT_RE = re.compile(r"\[\.\]|\(\.\)|\[dot\]|\(dot\)", re.I)


def refang(text):
    """Vuelve normales los enlaces "desactivados" (hxxps://sitio[.]com), para que las reglas de dominios
    los revisen. No cuenta como truco: así los escriben los equipos de seguridad al reenviar alertas."""
    if "xxp" in text.lower() or "[" in text or "(" in text:
        text = DEFANG_SCHEME_RE.sub(r"http\1\2", text)
        text = DEFANG_DOT_RE.sub(".", text)
    return text


def deobfuscate(text):
    """Devuelve (texto sin disfraces, trucos encontrados, un ejemplo de cada truco)."""
    if not text:
        return "", set(), {}
    text = refang(text)
    tricks, examples = set(), {}
    m = INVISIBLE_RE.search(text)
    if m:
        # Solo cuenta como truco si el invisible queda en medio de una palabra.
        if re.search(r"\w" + INVISIBLE_RE.pattern + r"+\w", text):
            tricks.add("invisibles")
            w = re.search(r"\w+" + INVISIBLE_RE.pattern + r"+\w+", text)
            examples["invisibles"] = INVISIBLE_RE.sub("·", w.group()) if w else ""
        text = INVISIBLE_RE.sub("", text)
    if FANCY_RE.search(text):
        tricks.add("letras_especiales")
        w = re.search(r"\S*" + FANCY_RE.pattern + r"\S*", text)
        examples["letras_especiales"] = w.group() if w else ""
        text = FANCY_RE.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)
    if HOMO_RE.search(text):
        text = re.sub(r"\S+", lambda m: _fix_homoglyphs(m.group(), tricks, examples), text)
    parts = URL_RE.split(text)
    # URL_RE tiene un grupo, así que split deja los enlaces en las posiciones impares.
    text = "".join(p if i % 2 else _fix_text(p, tricks, examples) for i, p in enumerate(parts))
    return text, tricks, examples
