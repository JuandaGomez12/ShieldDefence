"""Explicaciones para el usuario.

- build_prompt(): el mensaje que recibe el LLM (correo + resultado de la LSTM + señales).
- template_explanation(): explicación armada con plantillas. Se usa para crear
  los ejemplos del ajuste fino con LoRA y como respaldo si el LLM no está.
"""
import random
import re

from .signals import detect

SYSTEM = (
    "Eres ShieldAI, el asistente de ShieldDefence. Recibes un correo, el resultado de un "
    "clasificador LSTM y las señales encontradas en el texto. Explica en español sencillo, "
    "sin inventar datos que no estén en el correo, por qué el correo parece phishing o no, y "
    "da una recomendación práctica. Nunca contradigas la decisión del clasificador. "
    "Responde con este formato:\nExplicación: ...\nSeñales:\n- ...\nRecomendación: ..."
)


def level_of(prob: float, is_phishing: bool) -> str:
    if not is_phishing:
        return "legítimo"
    return "alto" if prob >= 0.85 else "medio"


def build_prompt(email: str, is_phishing: bool, prob: float, signals=None) -> str:
    signals = detect(email) if signals is None else signals
    verdict = "PHISHING" if is_phishing else "LEGÍTIMO"
    conf = prob if is_phishing else 1 - prob
    sig = "\n".join(f"- {s.title}: {s.evidence}" for s in signals) or "- (ninguna señal clara por reglas)"
    body = email.strip()
    if len(body) > 1800:
        body = body[:1800] + " [...]"
    return (
        f"Correo:\n<<<\n{body}\n>>>\n\n"
        f"Resultado del clasificador: {verdict} (confianza {conf:.0%})\n"
        f"Señales encontradas por reglas:\n{sig}"
    )


# ---------------- plantillas ----------------
_OPEN_PHISH = [
    "Este correo tiene varias características de un intento de phishing.",
    "El clasificador marcó este correo como phishing y hay detalles en el texto que lo respaldan.",
    "Hay que tener cuidado con este mensaje: tiene la forma típica de un correo de phishing.",
    "Este mensaje parece un intento de engaño para que entregues datos o dinero.",
]
_OPEN_PHISH_NOSIG = [
    "El clasificador marcó este correo como phishing, aunque no tiene las señales más obvias. "
    "Los correos hechos con IA suelen verse normales, así que vale la pena revisarlo con calma.",
    "Aunque el correo está bien escrito y no tiene errores, el modelo lo considera sospechoso por su "
    "forma y por lo que le pide al lector.",
]
_OPEN_LEGIT = [
    "Este correo no muestra señales claras de phishing.",
    "El clasificador considera que este correo es legítimo.",
    "No encontré nada en este mensaje que indique un intento de engaño.",
]
_REC_PHISH = [
    "No abras los enlaces ni respondas con datos personales. Si tienes dudas, entra tú mismo a la página o "
    "aplicación oficial de la entidad, o comunícate por sus canales oficiales.",
    "No hagas clic ni descargues nada. Si crees que puede ser real, verifica directamente con la entidad usando "
    "un número o página que ya conozcas, no los que vienen en el correo.",
    "Lo mejor es no responder y reportarlo como phishing en tu correo. Nunca compartas claves, códigos ni datos "
    "de tarjetas por este medio.",
]
_REC_MONEY = [
    "No envíes dinero ni compres tarjetas de regalo. Confirma la solicitud llamando directamente a la persona "
    "por un número que ya tengas.",
]
_REC_LEGIT = [
    "Puedes continuar con normalidad, pero si el correo te pide algo inesperado, confírmalo por otro medio.",
    "Parece seguro, aunque siempre es buena idea revisar el remitente antes de abrir enlaces o adjuntos.",
    "No hace falta hacer nada especial. Aun así, nunca compartas contraseñas o códigos aunque el correo parezca real.",
]
def _legit_points(email, signals):
    """Solo afirma lo que es cierto según las señales del correo."""
    keys = {s.key for s in signals}
    pts = []
    if not keys & {"credentials", "money_request"}:
        pts.append("No pide contraseñas, códigos, datos bancarios ni dinero.")
    if not keys & {"urgency", "threat", "deadline"}:
        pts.append("No presiona con plazos cortos ni amenazas.")
    if not re.search(r"https?://|www\.", email, re.I):
        pts.append("No trae enlaces para hacer clic.")
    if not pts:
        pts.append("El contenido es coherente con un mensaje normal de su tipo.")
    return pts


def template_explanation(email: str, is_phishing: bool, prob: float, signals=None, rng=None) -> dict:
    rng = rng or random.Random()
    signals = detect(email) if signals is None else signals
    if is_phishing:
        opening = rng.choice(_OPEN_PHISH if signals else _OPEN_PHISH_NOSIG)
        points = [f"{s.title}: {s.detail}" for s in sorted(signals, key=lambda s: -s.weight)[:5]]
        if not points:
            points = ["El estilo y lo que pide el correo se parecen a los correos de phishing con los que se entrenó el modelo."]
        rec = rng.choice(_REC_MONEY if any(s.key in ("money_request", "secrecy") for s in signals) else _REC_PHISH)
    else:
        opening = rng.choice(_OPEN_LEGIT) if not signals else (
            "El clasificador considera que este correo es legítimo, aunque tiene algunos detalles que conviene revisar.")
        pts = _legit_points(email, signals)
        points = rng.sample(pts, k=min(2, len(pts)))
        for s in sorted(signals, key=lambda s: -s.weight)[:2]:
            points.append(f"Tiene un detalle a revisar ({s.title.lower()}: \"{s.evidence}\"), pero el modelo no lo "
                          "considera suficiente para decir que es fraude.")
        rec = rng.choice(_REC_LEGIT)
    return {"explicacion": opening, "senales": points, "recomendacion": rec}


def format_answer(exp: dict) -> str:
    lines = [f"Explicación: {exp['explicacion']}", "Señales:"]
    lines += [f"- {p}" for p in exp["senales"]]
    lines.append(f"Recomendación: {exp['recomendacion']}")
    return "\n".join(lines)


def parse_answer(text: str):
    """Convierte la salida del LLM al formato que usa la interfaz. None si no sirve."""
    m_exp = re.search(r"Explicaci[oó]n:\s*(.+?)(?=\n\s*Se[ñn]ales:|\n\s*Recomendaci[oó]n:|\Z)", text, re.S | re.I)
    m_rec = re.search(r"Recomendaci[oó]n:\s*(.+)", text, re.S | re.I)
    m_sig = re.search(r"Se[ñn]ales:\s*(.+?)(?=\n\s*Recomendaci[oó]n:|\Z)", text, re.S | re.I)
    if not (m_exp and m_rec):
        return None
    sig = [re.sub(r"^\s*[-*•]\s*", "", l).strip() for l in (m_sig.group(1) if m_sig else "").splitlines()]
    return {
        "explicacion": m_exp.group(1).strip(),
        "senales": [s for s in sig if s][:6],
        "recomendacion": m_rec.group(1).strip().split("\n\n")[0],
    }
