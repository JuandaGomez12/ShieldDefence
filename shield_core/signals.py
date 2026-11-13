"""Señales de phishing que se pueden ver en el texto de un correo.

Es la evolución del motor de reglas del prototipo (script.js): las mismas
señales, ahora en español e inglés y con algunas nuevas. Estas reglas NO
deciden si el correo es phishing (eso lo hace la LSTM). Solo sirven para
señalarle al usuario partes concretas del correo, para construir los
ejemplos del ajuste fino del LLM y como respaldo si el LLM no está.
"""
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlparse

BRANDS = [
    "bancolombia", "davivienda", "nequi", "daviplata", "bbva", "banco de bogota", "paypal", "amazon",
    "netflix", "google", "microsoft", "outlook", "office 365", "apple", "icloud", "facebook",
    "instagram", "whatsapp", "mercadolibre", "mercado libre", "dhl", "fedex", "servientrega",
    "visa", "mastercard", "dian", "chase", "wells fargo", "bank of america", "ebay", "onedrive",
]
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "cutt.ly", "is.gd", "ow.ly", "rb.gy", "shorturl.at"}
ODD_TLDS = (".xyz", ".top", ".click", ".tk", ".ru", ".icu", ".live", ".info", ".work", ".cn", ".zip", ".biz")
SECURITY_WORDS = ("seguridad", "verificacion", "verifica", "verify", "secure", "login", "signin", "acceso",
                  "soporte", "support", "account", "cuenta", "update", "actualiza", "validar", "validate",
                  "confirm", "pagos", "payment", "billing", "banco", "bank", "portal", "nomina")

URL_RE = re.compile(r"(https?://[^\s<>\"')]+|www\.[^\s<>\"')]+)", re.I)


def norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


@dataclass
class Signal:
    key: str
    title: str
    detail: str
    weight: int
    evidence: str = ""


def _first(rx, text):
    m = re.search(rx, text)
    return m.group(0) if m else ""


def find_urls(text: str):
    return [u.rstrip(".,;:)]") for u in URL_RE.findall(text or "")]


def host_of(url: str) -> str:
    try:
        return (urlparse(url if url.lower().startswith("http") else "http://" + url).hostname or "").lower()
    except ValueError:
        return ""


def detect(text: str):
    """Devuelve la lista de señales encontradas en el correo."""
    t = norm(text or "")
    out = []

    def add(key, title, detail, weight, evidence=""):
        if not any(s.key == key for s in out):
            out.append(Signal(key, title, detail, weight, evidence))

    # ---- Enlaces ----
    for url in find_urls(text):
        host = host_of(url)
        if not host:
            continue
        base = host[4:] if host.startswith("www.") else host
        label = base.split(".")[0]
        if re.search(r"[a-z][0-9]|[0-9][a-z]", label.replace("-", "")):
            clean = label.replace("0", "o").replace("1", "l").replace("3", "e").replace("5", "s")
            brand = next((b for b in BRANDS if b.replace(" ", "") in clean), None)
            add("mixed_domain", "Dominio sospechoso",
                f"el enlace {base} cambia letras por números" + (f" para parecerse a {brand.title()}" if brand else "")
                + ", algo típico de los sitios falsos.", 30, base)
        brand_in_host = next((b for b in BRANDS if b.replace(" ", "") in base.replace("-", "")), None)
        if brand_in_host and ("-" in label or base.count(".") >= 2 or base.endswith(ODD_TLDS)):
            add("brand_lookalike", "Dominio que imita una marca",
                f"el enlace {base} usa el nombre de {brand_in_host.title()}, pero no es su dominio oficial.", 25, base)
        if "-" in label and any(w in base for w in SECURITY_WORDS):
            add("security_domain", "Dominio con palabras de verificación",
                f"el enlace {base} mezcla guiones con palabras como \"seguridad\" o \"login\", algo común en sitios falsos.", 15, base)
        if base in SHORTENERS:
            add("shortener", "Enlace acortado", f"el enlace {base} esconde a qué página lleva realmente.", 20, base)
        if base.endswith(ODD_TLDS):
            add("odd_tld", "Terminación de dominio poco habitual",
                f"la terminación de {base} se usa mucho en sitios fraudulentos.", 15, base)
        if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", base):
            add("ip_url", "Enlace con dirección IP", "el enlace apunta a una dirección IP y no a un dominio oficial.", 25, base)
        if url.lower().startswith("http://"):
            add("no_https", "Conexión no segura", "el enlace no usa HTTPS, así que la conexión no está cifrada.", 10, base)

    # ---- Texto (español e inglés) ----
    ev = _first(r"\b(24|48|72) ?(horas|hours|hrs)\b|inmediatamente|de inmediato|urgente|ahora mismo|ultimo aviso|"
                r"ultima oportunidad|expira|lo antes posible|pague ahora|paga ahora|immediate(ly)?|urgent(ly)?|"
                r"as soon as possible|within \d+ (hours|days)|final notice|act now|expires? (today|soon)|prompt(ly)? action", t)
    if ev:
        add("urgency", "Lenguaje de urgencia",
            f"presiona con prisa (\"{ev}\"), una técnica común para que la persona actúe sin pensar.", 20, ev)
    ev = _first(r"suspendid\w*|suspension|bloquead\w*|bloqueo|cancelad\w*|sera eliminad\w*|acceso restringido|"
                r"desactivad\w*|suspended|locked|terminated|deactivated|restricted|will be closed|disabled", t)
    if ev:
        add("threat", "Amenaza de suspensión",
            f"usa el miedo a perder la cuenta o el servicio (\"{ev}\") para presionar.", 20, ev)
    ev = _first(r"(verificar|verifique|confirmar|confirme|actualizar|actualice|ingresa\w*|ingrese) (sus|tus|su|tu) "
                r"(datos|identidad|cuenta|clave|contrasena|informacion)|contrasena|\bclave\b|\bpin\b|\bcvv\b|"
                r"numero de (la )?tarjeta|datos bancarios|cuenta bancaria|usuario y contrasena|"
                r"password|credentials|social security|bank account|card number|verify your (account|identity)|"
                r"(verify|confirm|update|validate) your (account|identity|details|information|payment|billing|transaction)", t)
    if ev:
        add("credentials", "Pide datos sensibles",
            f"pide información personal o bancaria (\"{ev}\"), algo que las entidades serias no piden por correo.", 20, ev)
    ev = _first(r"estimad[oa]\(?a?\)? (cliente|usuario)|querid[oa] (usuario|amigo|cliente)|apreciad[oa] cliente|"
                r"dear (customer|user|member|friend|sir|beneficiary|account holder)|valued customer", t)
    if ev:
        add("generic_greeting", "Saludo genérico",
            f"no usa el nombre de la persona (\"{ev}\"), algo normal en correos masivos fraudulentos.", 10, ev)
    ev = _first(r"has ganado|ha ganado|premio|sorteo|herencia|reembolso|devolucion|saldo a favor|regalo gratis|"
                r"you (have )?won|winner|lottery|prize|inheritance|refund|compensation|claim your", t)
    if ev:
        add("prize", "Promesa de dinero o premio",
            f"ofrece dinero o premios inesperados (\"{ev}\") para llamar la atención.", 15, ev)
    ev = _first(r"tarjetas? de regalo|gift ?cards?|transferi\w*|consigna\w*|western union|moneygram|bitcoin|"
                r"criptomoneda\w*|wire transfer|send (the )?money|bank transfer|ganancias? .{0,20}garantizad\w*|"
                r"guaranteed (returns?|profit)|million (dollars|usd)|\$ ?\d[\d.,]* (million|millones)", t)
    if ev:
        add("money_request", "Pide mover dinero",
            f"busca que la persona envíe dinero, compre algo o invierta (\"{ev}\").", 20, ev)
    ev = _first(r"(te|le|les) (comparto|compartio|ha compartido|envio) .{0,25}?(archivo|documento|formato|enlace|factura)|"
                r"documento compartido|shared (a )?(file|document) with you|has shared|view (the )?document|"
                r"inicia sesion con tu cuenta|sign in to (view|access)", t)
    if ev and find_urls(text):
        add("shared_doc", "Documento compartido con enlace externo",
            f"invita a abrir un archivo en un enlace externo (\"{ev}\"), una forma común de robar contraseñas.", 15, ev)
    ev = _first(r"no (puedo|podre) (hablar|contestar)|estoy en una reunion|que quede entre nosotros|confidencial|"
                r"keep (this )?(confidential|between us)|i am in a meeting|can'?t talk", t)
    if ev:
        add("secrecy", "Pide discreción o evita hablar",
            f"evita que la persona confirme por otro medio (\"{ev}\"), típico del fraude del jefe.", 15, ev)
    ev = _first(r"antes del? (lunes|martes|miercoles|jueves|viernes|sabado|domingo|\d{1,2})\b|"
                r"(before|by) (monday|tuesday|wednesday|thursday|friday|tomorrow|end of (the )?day)", t)
    if ev and find_urls(text):
        add("deadline", "Pone una fecha límite",
            f"pide hacer algo antes de una fecha (\"{ev}\") junto con un enlace, para que no lo pienses mucho.", 10, ev)
    ev = _first(r"haga clic|haz clic|click (here|below|on the (following )?link|the link)|ingrese al siguiente enlace|"
                r"ingresa al enlace|siguiente enlace|follow the link|revisalo (aqui|en)", t)
    if ev and find_urls(text):
        add("click", "Pide hacer clic en un enlace", f"insiste en abrir un enlace (\"{ev}\").", 5, ev)
    return out


def rule_score(signals) -> int:
    return min(98, sum(s.weight for s in signals))
