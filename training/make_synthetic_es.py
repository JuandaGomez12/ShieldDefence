"""Genera correos sintéticos en español para entrenar (no para evaluar).

El español del dataset original es traducción automática de correos en inglés, y casi no tiene
correos normales del día a día en Colombia (universidad, trabajo, familia, avisos reales de bancos).
Por eso el modelo marcaba como phishing correos legítimos solo por estar en español.

Este script arma correos a partir de plantillas con muchas piezas intercambiables. Las dos clases
comparten saludos, firmas, nombres y largo, para que el modelo no pueda separarlas por el estilo.
Incluye casos difíciles de los dos lados:
  - legítimos que mencionan claves o códigos (para decir que nunca se piden), cobros reales, enlaces oficiales
  - phishing sin enlaces (fraude del jefe, cambio de cuenta de un proveedor, pedir el código por chat)

La prueba en español (data/spanish_eval.csv) se escribió aparte, a mano, y no sale de aquí.

Uso: python training/make_synthetic_es.py --n 2500 --out data/synthetic_es
"""
import argparse
import random
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shield_core.preprocess import for_model  # noqa: E402

R = random.Random()

NOMBRES = ["Juan", "María", "Camila", "Andrés", "Laura", "Santiago", "Valentina", "Daniel", "Natalia", "Felipe",
           "Carolina", "Sebastián", "Paula", "Diego", "Juliana", "Mateo", "Sara", "Alejandro", "Daniela", "Carlos",
           "Ana", "Jorge", "Luisa", "Ricardo", "Mónica", "Esteban", "Catalina", "Óscar", "Adriana", "Nicolás"]
APELLIDOS = ["Gómez", "Rodríguez", "Martínez", "López", "García", "Pérez", "Ramírez", "Torres", "Herrera", "Díaz",
             "Moreno", "Rojas", "Vargas", "Castro", "Ortiz", "Jiménez", "Suárez", "Cárdenas", "Mejía", "Cuéllar"]
EMPRESAS = ["Inversiones Andinas", "Logística del Norte", "Grupo Solarte", "Constructora Calle 80", "TecnoSoluciones",
            "Agro Boyacá", "Distribuidora La Esperanza", "Consultores Asociados", "Clínica San Rafael", "Editorial Faro"]
BANCOS = ["Bancolombia", "Davivienda", "BBVA", "Banco de Bogotá", "Nequi", "Daviplata", "Banco Popular",
          "Banco de Occidente", "Scotiabank Colpatria", "Banco Caja Social"]
BANCO_DOM = {"Bancolombia": "bancolombia.com", "Davivienda": "davivienda.com", "BBVA": "bbva.com.co",
             "Banco de Bogotá": "bancodebogota.com", "Nequi": "nequi.com.co", "Daviplata": "daviplata.com",
             "Banco Popular": "bancopopular.com.co", "Banco de Occidente": "bancodeoccidente.com.co",
             "Scotiabank Colpatria": "scotiabankcolpatria.com", "Banco Caja Social": "bancocajasocial.com"}
TIENDAS = ["Mercado Libre", "Falabella", "Éxito", "Amazon", "Rappi", "Alkosto", "Homecenter", "Linio"]
SERVICIOS = ["Netflix", "Spotify", "Claro", "Movistar", "Tigo", "ETB", "Enel", "Vanti", "Acueducto de Bogotá"]
MATERIAS = ["Cálculo Diferencial", "Programación Orientada a Objetos", "Probabilidad y Estadística",
            "Bases de Datos", "Inteligencia Artificial", "Física Mecánica", "Arquitectura de Software",
            "Ecuaciones Diferenciales", "Redes de Computadores", "Procesamiento de Imágenes"]
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]
CIUDADES = ["Bogotá", "Medellín", "Cali", "Barranquilla", "Bucaramanga", "Pereira", "Manizales", "Tunja"]
LUGARES = ["el salón B-201", "el bloque E, salón 304", "la sala de juntas del piso 3", "el auditorio principal",
           "la biblioteca", "el laboratorio de redes", "Teams", "Meet", "Zoom", "la cafetería"]
SALUDOS = ["Hola", "Hola, buenos días", "Buenos días", "Buenas tardes", "Buenas noches", "Hola a todos",
           "Hola equipo", "Hola {n}", "Buen día, {n}", "Estimado {n}", "Estimada {n}", "Cordial saludo",
           "Estimados estudiantes", "Hola, ¿cómo estás?", "Qué tal, {n}", "Señor(a) usuario(a)",
           "Apreciado cliente", "Hola, buenas", "Muy buenos días", "Querida {n}", ""]
DESPEDIDAS = ["Saludos", "Gracias", "Muchas gracias", "Cordialmente", "Un abrazo", "Quedo atento", "Quedo atenta",
              "Feliz día", "Atentamente", "Mil gracias", "Quedo pendiente", "Bendiciones", "Hasta pronto", ""]
EXTRAS = ["", "", "", "Enviado desde mi iPhone", "Enviado desde mi celular", "Este mensaje es confidencial.",
          "Por favor no imprimas este correo si no es necesario."]

DOM_FALSOS = ["{b}-seguridad.xyz", "{b}-verificacion.top", "{b}.cuenta-segura.info", "{b}-soporte.live",
              "{b}1.com", "{b}-validacion.click", "{b}0nline.com", "secure-{b}.net", "{b}-co.app", "{b}.web-acceso.com",
              "acceso{b}.site", "{b}-clientes.co"]
ACORTADORES = ["bit.ly/{c}", "tinyurl.com/{c}", "cutt.ly/{c}", "rb.gy/{c}", "is.gd/{c}"]


def nombre():
    return R.choice(NOMBRES)


def persona():
    return f"{R.choice(NOMBRES)} {R.choice(APELLIDOS)}"


def fecha():
    return f"{R.choice(DIAS)} {R.randint(1, 28)} de {R.choice(MESES)}"


def hora():
    return R.choice([f"{R.randint(7, 11)}:{R.choice(['00', '30'])} a. m.", f"{R.randint(1, 6)}:{R.choice(['00', '30'])} p. m.",
                     f"{R.randint(7, 18)}:{R.choice(['00', '15', '30', '45'])}"])


def plata():
    v = R.choice([R.randint(2, 99) * 1000, R.randint(1, 30) * 50000, R.randint(1, 9) * 100000 + R.randint(0, 9) * 1000])
    return "$" + f"{v:,}".replace(",", ".")


def codigo(n=6):
    return "".join(R.choice("0123456789") for _ in range(n))


def slug(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if c.isalnum())


def url_falsa(marca):
    b = slug(marca)
    r = R.random()
    if r < 0.15:
        return "https://" + R.choice(ACORTADORES).format(c="".join(R.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(6)))
    if r < 0.25:
        return f"http://{R.randint(23, 213)}.{R.randint(0, 255)}.{R.randint(0, 255)}.{R.randint(1, 254)}/{b}"
    dom = R.choice(DOM_FALSOS).format(b=b)
    return f"{R.choice(['http://', 'https://', 'http://www.', 'https://'])}{dom}/{R.choice(['login', 'acceso', 'verificar', 'clientes', 'actualizar', 'seguridad', 'validar'])}"


def url_real(dom):
    return R.choice([f"https://www.{dom}", f"https://{dom}", f"www.{dom}"])


def armar(cuerpo, saludo_n=None, firma=None):
    """Une saludo, párrafos, despedida y firma, con variaciones de formato."""
    n = saludo_n or nombre()
    s = R.choice(SALUDOS).format(n=n)
    partes = []
    if s:
        partes.append(s + R.choice([",", ".", ":", "", ","]))
    partes.extend(p for p in cuerpo if p)
    d = R.choice(DESPEDIDAS)
    if d:
        partes.append(d + R.choice([",", ".", ""]) + ("\n" + firma if firma else ""))
    elif firma:
        partes.append(firma)
    extra = R.choice(EXTRAS)
    if extra:
        partes.append(extra)
    sep = R.choice(["\n\n", "\n", "\n\n", " "])
    texto = sep.join(partes)
    r = R.random()
    if r < 0.12:  # mucha gente escribe sin tildes
        texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
    elif r < 0.17:
        texto = texto.lower()
    if R.random() < 0.3:
        asunto = cuerpo[0][:R.randint(25, 60)].split(".")[0]
        texto = f"Asunto: {asunto}\n\n{texto}" if R.random() < 0.5 else f"{asunto}\n{texto}"
    return texto


# ---------------------------------------------------------------- legítimos
def l_profesor():
    m = R.choice(MATERIAS)
    op = R.choice([
        [f"El {R.choice(['taller', 'proyecto', 'informe', 'laboratorio'])} de {m} se entrega el {fecha()} antes de las {hora()}.",
         R.choice(["Usen la plantilla que está en la carpeta del curso.",
                   "El documento debe incluir portada, resultados y conclusiones.", "No se reciben entregas tarde."]),
         "Si tienen dudas, me pueden escribir por este medio o en el horario de atención."],
        [f"La clase de {m} del {fecha()} será en {R.choice(LUGARES)}.", "Traigan calculadora y la guía impresa.",
         R.choice(["Gracias por la comprensión.", "Cualquier inquietud me avisan.", ""])],
        [f"Subí las calificaciones del {R.choice(['primer', 'segundo', 'tercer'])} examen de {m}.",
         "Quedaron en la plataforma, en la pestaña de calificaciones.",
         f"Si quieren revisar el examen, los espero el {fecha()} a las {hora()} en mi oficina."],
        [f"Comparto las diapositivas de la clase de hoy sobre {R.choice(['redes neuronales', 'normalización', 'integrales', 'árboles de decisión', 'concurrencia'])}.",
         "Quedaron en la carpeta del curso.", f"Para la próxima clase lean el capítulo {R.randint(2, 12)} del libro."],
    ])
    return armar(op, firma=R.choice(["Profesor de la asignatura", persona(), f"Prof. {persona()}", "Monitor del curso", ""]))


def l_estudiante():
    m = R.choice(MATERIAS)
    op = R.choice([
        [f"Profe, ¿el quiz de {m} entra el tema de la última clase?", "Es que no alcancé a tomar apuntes.", "Muchas gracias por su tiempo."],
        [f"Quería saber si es posible presentar el quiz de {m} otro día, porque tuve una cita médica.", "Adjunto la incapacidad."],
        [f"Chicos, ¿nos vemos el {fecha()} en {R.choice(LUGARES)} para avanzar el proyecto de {m}?",
         R.choice(["Yo llevo lo de la base de datos.", "Cada uno revise su parte antes.", "Avisen si no pueden."])],
        ["Hice un resumen de lo que hablamos hoy.", f"A mí me toca la introducción y a {nombre()} los resultados.",
         "Si alguien no puede el martes, avise para mover la reunión."],
    ])
    return armar(op, firma=nombre())


def l_universidad():
    op = R.choice([
        [f"La inscripción de asignaturas para el próximo semestre abre el {fecha()}.",
         "Recuerde revisar su horario de inscripción en el sistema académico con su usuario institucional.",
         "Nunca le pediremos su contraseña por correo."],
        [f"El {fecha()} habrá mantenimiento programado del correo institucional de {hora()} a {hora()}.",
         "Durante ese tiempo el servicio puede presentar intermitencias. No es necesario que haga nada."],
        [f"Lo invitamos a la charla '{R.choice(['Mujeres en ingeniería', 'IA responsable', 'Emprendimiento tecnológico', 'Ciberseguridad para todos'])}' el {fecha()} en {R.choice(LUGARES)}.",
         "La entrada es libre.", f"Más información en {url_real('escuelaing.edu.co')}"],
        ["Le recordamos que el plazo para el pago de la matrícula vence el " + fecha() + ".",
         "Puede descargar el recibo desde el portal de estudiantes o pagar en cualquier sucursal autorizada."],
    ])
    return armar(op, firma=R.choice(["Oficina de Registro Académico", "Bienestar Universitario", "Mesa de Servicios TI", "Decanatura"]))


def l_trabajo():
    e = R.choice(EMPRESAS)
    op = R.choice([
        [f"Les recuerdo la reunión del {fecha()} a las {hora()} en {R.choice(LUGARES)}.",
         R.choice(["Vamos a revisar el avance del proyecto.", "Traigan las cifras del mes.", "La agenda va adjunta."])],
        [f"Adjunto el informe de {R.choice(['ventas', 'gastos', 'avance', 'inventario'])} de {R.choice(MESES)}.",
         R.choice(["Si ven algún error me avisan.", "Quedo atento a sus comentarios.", "Lo revisamos en el comité."])],
        [f"Les informo que estaré de vacaciones del {fecha()} al {fecha()}.", f"Cualquier tema urgente lo pueden ver con {persona()}."],
        [f"Recursos Humanos informa que la nómina de {R.choice(MESES)} se pagó hoy.",
         "Pueden consultar el desprendible en el portal de empleados. Si encuentran alguna novedad, escriban a nómina."],
        [f"Bienvenido al equipo de {e}.", f"Tu primer día es el {fecha()}; te esperamos a las {hora()} en recepción.",
         "Trae tu documento de identidad para la carnetización."],
        [f"{nombre()}, ¿me puedes enviar la cotización que te mandó el proveedor?", "La necesito para el comité del " + fecha() + "."],
    ])
    return armar(op, firma=R.choice([persona(), f"{persona()}\n{e}", "Gerencia", "Recursos Humanos", ""]))


def l_familia():
    op = R.choice([
        [f"¿Vienes el {fecha()} a almorzar donde la abuela?", R.choice(["Ella va a hacer ajiaco.", "Avísame para comprar más comida.", "Trae a los niños."])],
        [f"Te mando las fotos del paseo a {R.choice(CIUDADES)}.", "Quedaron muy bonitas, especialmente las del atardecer."],
        [f"Feliz cumpleaños, {nombre()}!", "Que tengas un día muy especial. Te queremos mucho."],
        ["Ya llegué bien a la casa.", "Mañana te llamo para contarte cómo me fue."],
        [f"¿Alguien tiene el número del plomero?", "Se dañó otra vez la llave del baño."],
    ])
    return armar(op, firma=R.choice([nombre(), "Mamá", "Tu hermano", "Papá", ""]))


def l_banco():
    b = R.choice(BANCOS)
    op = R.choice([
        [f"{b} te informa que realizaste una compra por {plata()} en {R.choice(TIENDAS)} el {fecha()} a las {hora()}.",
         "Si no reconoces esta transacción, comunícate con nuestra línea de atención que aparece al respaldo de tu tarjeta."],
        [f"Tu extracto de {R.choice(MESES)} ya está disponible en la app de {b}.",
         "Recuerda: nunca te pediremos tu clave, tu token ni códigos por correo, llamada o mensaje."],
        [f"Recibiste una transferencia por {plata()} de {persona()}.", f"Puedes ver el detalle en la app de {b}."],
        [f"Tu clave de {b} fue cambiada exitosamente el {fecha()}.",
         "Si no fuiste tú, llama de inmediato a la línea oficial. Nunca compartas tus claves."],
        [f"En {b} trabajamos por tu seguridad.", "Te recordamos que nunca te enviaremos enlaces para actualizar tus datos.",
         f"Ingresa siempre escribiendo {url_real(BANCO_DOM[b])} en tu navegador."],
    ])
    return armar(op, firma=b)


def l_codigo():
    s = R.choice(BANCOS + SERVICIOS + ["Google", "Microsoft", "WhatsApp", "Mercado Libre"])
    op = [f"Tu código de verificación de {s} es {codigo()}.", R.choice([
        "No lo compartas con nadie. Nadie de nuestro equipo te lo va a pedir.",
        "Este código vence en 10 minutos.", "Si no solicitaste este código, ignora este mensaje."])]
    return armar(op, firma=s)


def l_compra():
    t = R.choice(TIENDAS)
    op = R.choice([
        [f"Tu pedido #{codigo(8)} de {t} fue enviado.", f"Llegará entre el {fecha()} y el {fecha()}.",
         f"Número de guía: {codigo(10)} de {R.choice(['Servientrega', 'Interrapidísimo', 'Coordinadora', 'Envía'])}."],
        [f"Gracias por tu compra en {t} por {plata()}.", "Adjuntamos la factura electrónica de tu compra."],
        [f"Tu factura de {R.choice(SERVICIOS)} del mes de {R.choice(MESES)} por {plata()} ya está disponible.",
         "Puedes pagarla en los puntos autorizados o desde la app oficial antes de la fecha de vencimiento."],
        [f"Tu suscripción a {R.choice(SERVICIOS)} se renovó el {fecha()}.", f"Se hizo un cobro de {plata()} a tu medio de pago registrado."],
    ])
    return armar(op, firma=R.choice([t, "Servicio al cliente", ""]))


def l_cita():
    op = R.choice([
        [f"Tiene programada una consulta de {R.choice(['medicina general', 'odontología', 'optometría', 'laboratorio'])} para el {fecha()}, {hora()}.",
         "Preséntese con anticipación y traiga su documento.", "Si necesita reprogramar, use la app de la EPS."],
        [f"Su resultado de laboratorio ya está disponible.", "Puede consultarlo en la app de la EPS con su usuario."],
    ])
    return armar(op, firma=R.choice(["Clínica San Rafael", "EPS Sanitas", "Compensar", "Sura"]))


def l_boletin():
    op = [R.choice([f"Esta semana en nuestro boletín: {R.choice(['nuevos cursos', 'eventos de la ciudad', 'recetas fáciles', 'tecnología'])}.",
                    "Te compartimos las noticias más importantes del mes.", "Estas son las ofertas de la temporada."]),
          R.choice(["Si no quieres recibir más correos, puedes darte de baja al final de este mensaje.",
                    "Gracias por ser parte de nuestra comunidad.", ""])]
    return armar(op, firma=R.choice(EMPRESAS + TIENDAS))


LEGIT = [l_profesor, l_estudiante, l_universidad, l_trabajo, l_familia, l_banco, l_codigo, l_compra, l_cita, l_boletin]
LEGIT_W = [12, 10, 8, 14, 8, 10, 6, 10, 5, 7]


# ---------------------------------------------------------------- phishing
def p_banco():
    b = R.choice(BANCOS)
    op = R.choice([
        [f"Detectamos un acceso inusual a su cuenta de {b}.", R.choice([
            "Por su seguridad la cuenta fue bloqueada temporalmente.", "Su cuenta será suspendida si no confirma su identidad."]),
         f"Ingrese sus datos en {url_falsa(b)} {R.choice(['en las próximas 24 horas', 'hoy mismo', 'antes de las 6 p. m.', 'lo antes posible'])}."],
        [f"Su tarjeta de {b} tiene un cobro pendiente de {plata()}.", "Para cancelarlo verifique su usuario y clave en el siguiente enlace:", url_falsa(b)],
        [f"{b} le informa que tiene puntos por vencer.", f"Redímalos ahora ingresando con su usuario y clave en {url_falsa(b)}."],
        [f"Hemos actualizado nuestros términos de seguridad.", f"Todos los clientes de {b} deben actualizar sus datos.",
         f"Haga clic aquí: {url_falsa(b)}", "Si no lo hace, su cuenta quedará inactiva."],
    ])
    return armar(op, firma=R.choice([b, f"Departamento de seguridad {b}", "Servicio al cliente", ""]))


def p_institucional():
    op = R.choice([
        ["Su buzón de correo institucional alcanzó el límite de almacenamiento.", "Para no perder sus mensajes, valide su cuenta aquí:",
         url_falsa(R.choice(["escuelaing", "outlook", "office365", "correo"]))],
        ["Detectamos actividad inusual en su cuenta de estudiante.",
         f"Para evitar el cierre, verifique su usuario y contraseña en {url_falsa(R.choice(['escuelaing', 'microsoft', 'portal']))}."],
        ["La universidad está actualizando los datos de los estudiantes para el próximo semestre.",
         f"Complete el formulario con su usuario, contraseña y número de documento: {url_falsa('formulario')}"],
    ])
    return armar(op, firma=R.choice(["Mesa de ayuda", "Soporte TI", "Administrador del sistema", "Registro Académico"]))


def p_jefe():
    """Fraude del jefe: casi nunca tiene enlaces, lo que pide es dinero, códigos o discreción."""
    n = nombre()
    op = R.choice([
        [R.choice(["Ando en una junta toda la tarde y no me puedo conectar.", "Estoy en un congreso y no puedo contestar llamadas.", "Tengo el celular sin señal."]),
         f"Hazme el favor de conseguir {R.randint(2, 6)} tarjetas prepago de {plata()} {R.choice(['de Google Play', 'de Amazon', 'del Éxito', 'de Apple'])}.",
         "Raspa el código y me lo mandas en foto. Mañana te reintegro."],
        ["¿Estás en la oficina?", f"Necesito que hagas una transferencia urgente de {plata()} a un proveedor nuevo.",
         "Te paso los datos de la cuenta. Que quede entre nosotros por ahora, es un tema confidencial."],
        [f"{n}, necesito un favor rápido y discreto.", "No le comentes a nadie todavía.",
         f"Hay que pagar hoy una factura de {plata()} y yo no tengo acceso al banco. Te envío la cuenta por WhatsApp."],
        ["Te escribo desde mi correo personal porque el corporativo está fallando.",
         f"Por favor cambia la cuenta de nómina de {persona()} por una nueva que te mando ahora. Es urgente para el pago de hoy."],
    ])
    return armar(op, saludo_n=n, firma=R.choice(["El gerente", "Enviado desde mi iPhone", persona(), "Director", ""]))


def p_proveedor():
    e = R.choice(EMPRESAS)
    op = R.choice([
        [f"Les informamos que {e} cambió de cuenta bancaria.", f"A partir de hoy los pagos de las facturas pendientes por {plata()} deben hacerse a la nueva cuenta.",
         "Por favor no usen la cuenta anterior. Confirmen cuando hagan la consignación."],
        [f"Adjuntamos la factura vencida número {codigo(5)} por {plata()}.", "Si no se paga hoy se reportará a centrales de riesgo.",
         f"Para evitar cobros jurídicos comuníquese al {R.choice(['3', '6'])}{codigo(9)}."],
    ])
    return armar(op, firma=R.choice([e, f"Cartera {e}", persona()]))


def p_codigo():
    s = R.choice(["WhatsApp", "Nequi", "Daviplata", "tu banco", "Mercado Libre"])
    op = R.choice([
        [f"Hola, por error te llegó un código de {s} que era para mí.", "¿Me lo puedes reenviar por favor? Es urgente."],
        [f"Somos del área de seguridad de {s}.", "En unos minutos le llegará un código por mensaje de texto.",
         "Por favor díganos el código para cancelar una compra sospechosa."],
    ])
    return armar(op, firma=R.choice([nombre(), f"Seguridad {s}", ""]))


def p_paquete():
    t = R.choice(["Servientrega", "DHL", "FedEx", "4-72", "Interrapidísimo", "Coordinadora"])
    op = [f"Su paquete está retenido en nuestra bodega de {R.choice(CIUDADES)}.",
          f"Para liberarlo debe pagar un arancel de {plata()}.",
          f"Realice el pago en {url_falsa(t)} {R.choice(['antes de 48 horas', 'hoy', 'o será devuelto al remitente'])}."]
    return armar(op, firma=t)


def p_premio():
    op = R.choice([
        [f"¡Felicitaciones! Usted ha ganado {plata()} en el sorteo de {R.choice(TIENDAS)}.",
         f"Para reclamar su premio ingrese sus datos bancarios en {url_falsa(R.choice(TIENDAS))}."],
        ["Usted fue seleccionado como beneficiario de un subsidio del gobierno.",
         f"Para recibir {plata()} complete el registro con su número de cuenta en {url_falsa('subsidios')}."],
        [f"Tiene un saldo a favor de {plata()} por devolución de impuestos.",
         f"Reclámelo antes del {fecha()} en {url_falsa('dian')}."],
    ])
    return armar(op, firma=R.choice(["Departamento de premios", "DIAN", "Gobierno Nacional", ""]))


def p_inversion():
    op = [R.choice([f"Invierta {plata()} y reciba el doble en 30 días.", "Ganancias garantizadas con nuestra plataforma de criptomonedas.",
                    "Esta es una oportunidad única de inversión con rendimientos del 20% mensual."]),
          R.choice(["Cupos limitados, solo hasta hoy.", "Ya hay cientos de colombianos ganando desde la casa.", "No necesita experiencia."]),
          R.choice(["Escríbame por WhatsApp para enviarle los datos de la consignación.", f"Regístrese en {url_falsa('inversiones')}", "Responda este correo para más información."])]
    return armar(op, firma=R.choice([persona(), "Asesor financiero", ""]))


def p_servicio():
    s = R.choice(SERVICIOS)
    op = [f"No pudimos procesar el pago de su suscripción a {s}.", "Su servicio será suspendido.",
          f"Actualice su método de pago aquí: {url_falsa(s)}"]
    return armar(op, firma=s)


def p_documento():
    op = [f"{persona()} te compartió un documento: '{R.choice(['Factura', 'Nómina', 'Contrato', 'Propuesta', 'Cotización'])} {R.choice(MESES)}.pdf'.",
          "Para verlo inicia sesión con tu cuenta de correo.", url_falsa(R.choice(["onedrive", "sharepoint", "drive", "docs"]))]
    return armar(op, firma=R.choice(["OneDrive", "Google Drive", "SharePoint", ""]))


def p_empleo():
    op = [f"Fue seleccionado para un trabajo desde casa con un pago de {plata()} semanales.",
          "Para empezar debe enviar una copia de su cédula y pagar el kit de inscripción de " + plata() + ".",
          R.choice(["Consigne a la cuenta que le enviaremos.", f"Inscríbase en {url_falsa('empleos')}"])]
    return armar(op, firma=R.choice(["Recursos Humanos", persona(), ""]))


def p_sutil():
    """Phishing bien escrito, sin urgencia exagerada: se parece mucho a un correo normal."""
    b = R.choice(BANCOS + ["la universidad", "Recursos Humanos"])
    op = R.choice([
        [f"Como parte de la actualización anual, {b} está verificando la información de contacto de sus usuarios.",
         "El proceso toma menos de dos minutos.", f"Puede completarlo en {url_falsa(b)}", "Gracias por ayudarnos a mantener sus datos al día."],
        [f"Te escribo de parte de {persona()}, del área de nómina.", "Estamos migrando al nuevo sistema de pagos.",
         f"Para que tu pago de {R.choice(MESES)} no se retrase, confirma tus datos bancarios en {url_falsa('nomina')}."],
        ["Le compartimos el nuevo reglamento interno, que debe leer y firmar esta semana.",
         f"Ingrese con su correo y contraseña en {url_falsa('portal')} para firmarlo."],
    ])
    return armar(op, firma=R.choice([persona(), b if isinstance(b, str) else "", ""]))


PHISH = [p_banco, p_institucional, p_jefe, p_proveedor, p_codigo, p_paquete, p_premio, p_inversion, p_servicio,
         p_documento, p_empleo, p_sutil]
PHISH_W = [14, 8, 12, 7, 6, 8, 7, 6, 7, 7, 5, 13]


OFICIALES = ["escuelaing.edu.co", "unal.edu.co", "mercadolibre.com.co", "falabella.com.co", "netflix.com",
             "dian.gov.co", "sura.com", "compensar.com", "eltiempo.com", "github.com", "zoom.us", "teams.microsoft.com",
             "docs.google.com", "servientrega.com"] + list(BANCO_DOM.values())


def con_enlace(texto):
    """Una parte de los legítimos lleva un enlace oficial, como pasa en la vida real."""
    linea = R.choice(["Más información en {u}", "Enlace de la reunión: {u}", "Puedes ver el detalle en {u}", "{u}",
                      "Consulta aquí: {u}", "Ver en el navegador: {u}"]).format(u=url_real(R.choice(OFICIALES)) + R.choice(["", "/", "/ayuda", "/eventos", "/pedidos"]))
    partes = texto.split("\n\n")
    partes.insert(max(1, len(partes) - 1), linea)
    return "\n\n".join(partes)


# ---------------------------------------------------------------- más legítimos del día a día
# En el dataset original casi no hay correos personales legítimos, y el 40% del phishing son estafas
# nigerianas escritas en tono personal y hablando de plata. Por eso el modelo veía "personal + plata"
# como estafa. Estos generadores combinan frases sueltas para dar mucha variedad.
COMIDAS = ["ajiaco", "bandeja paisa", "sancocho", "lasaña", "arepas", "tamales", "pastel de pollo", "torta de chocolate",
           "galletas de avena", "arroz con pollo", "empanadas", "pan de bono", "natilla", "buñuelos"]
FAMILIA = ["mamá", "papá", "tía", "tío", "abuela", "abuelo", "prima", "primo", "hermana", "hermano", "mi amor", "suegra", "madrina"]


def frases(*grupos):
    """Elige una frase de cada grupo (los grupos vacíos o None se saltan)."""
    return [R.choice(g) for g in grupos if g]


def l_plata_familia():
    """Plata entre familia y amigos: transferencias reales, cuotas, préstamos devueltos."""
    q = plata()
    return armar(frases(
        [f"Ahí va mi parte de la vaca para el asado: {q}.", f"Te consigné {q} por Nequi, lo del almuerzo del domingo.",
         f"Ahí te devolví los {q} que me prestaste la semana pasada.", f"Mañana te paso los {q} del arriendo, ya me pagaron.",
         f"Te mandé {q} para el mercado de la semana.", f"Recogimos {q} entre todos para el regalo de grado de {nombre()}.",
         f"Le pagué a la señora del aseo los {q} de este mes.", f"Ya quedó pago el recibo del gas, fueron {q}."],
        ["¿Me confirmas si te llegó?", "Avísame cuando lo veas.", "Cualquier cosa me dices.", "Revisa y me cuentas.", ""],
        ["Gracias por todo.", "Te quiero mucho.", "Un abrazo grande.", "Nos vemos el finde.", ""]), firma=R.choice([nombre(), "Mamá", "Tu hija", ""]))


def l_personal():
    return armar(frases(
        [f"Me pasaron una receta de {R.choice(COMIDAS)} buenísima, te la copio abajo.", f"¿Vienes el {fecha()} a la casa? Vamos a hacer {R.choice(COMIDAS)}.",
         "Ya llegamos bien del viaje, el vuelo estuvo tranquilo.", f"Te cuento que {nombre()} pasó a la universidad.",
         f"Feliz cumpleaños, {nombre()}! Que la pases muy bonito.", "Lamento mucho lo de tu papá, cuenta conmigo para lo que necesites.",
         f"¿Ya viste las fotos del matrimonio de {nombre()}? Quedaron hermosas.", "El perro ya está mejor, el veterinario dijo que era solo una infección.",
         f"Nos vemos el {fecha()} en {R.choice(['el centro comercial', 'la casa de la abuela', 'el parque', 'la finca'])}.",
         f"¿Alguien sabe a qué hora empieza la novena del {fecha()}?", "Se me quedó la sombrilla en tu carro, la recojo mañana."],
        [f"Se cocina a fuego lento como {R.randint(20, 90)} minutos y se sirve caliente.", "Lleva algo de tomar si puedes.",
         "Ya le avisé a los demás.", "Ojalá puedas venir.", "Me cuentas cómo te va.", "Te extrañamos mucho.", ""],
        ["Un abrazo.", "Besos.", "Cuídate mucho.", "Hablamos luego.", ""]), firma=R.choice([nombre(), "Tu tía", "Mamá", ""]))


def l_comunidad():
    return armar(frases(
        [f"El {fecha()} habrá corte de agua en el sector de {hora()} a {hora()}.", "Recuerden que la cuota de administración vence el 10 de cada mes.",
         f"La reunión de padres del colegio quedó para el {fecha()} a las {hora()}.", f"La profesora de {nombre()} pide traer el material de artes el lunes.",
         f"El grupo de oración se reúne el {fecha()} en la parroquia.", "Se perdió un gato gris en la torre 3; si lo ven, avísenle a la portería."],
        ["Gracias por su colaboración.", "Cualquier duda, en la administración.", ""]),
        firma=R.choice(["Administración", "Consejo de administración", "Junta de Acción Comunal", "Coordinación académica", nombre()]))


def l_entidad():
    """Avisos reales de entidades, sin pedir datos ni traer enlaces raros."""
    return armar(frases(
        [f"La jornada de vacunación contra la influenza será el {fecha()} en todas las sedes.",
         f"Su solicitud quedó radicada con el número {codigo(7)}; la respuesta llega en máximo 15 días hábiles.",
         "Le recordamos que la renovación del pasaporte se hace solo con cita previa.",
         f"El impuesto predial con descuento se puede pagar hasta el {fecha()} en los bancos autorizados.",
         f"La licencia de conducción que vence este año se renueva en los centros autorizados con examen médico.",
         f"Su afiliación a la caja de compensación quedó activa desde el {fecha()}."],
        ["Si tiene dudas, comuníquese con nuestras líneas de atención.", "Lleve su documento de identidad.", "No es necesario realizar ningún otro trámite.", ""]),
        firma=R.choice(["EPS Sanitas", "Registraduría Nacional", "DIAN", "Secretaría de Movilidad", "Cancillería", "Compensar"]))


def l_seguridad_interna():
    """Campañas internas legítimas que hablan de contraseñas y phishing (pero no piden nada)."""
    return armar(frases(
        ["Este mes haremos un simulacro de correos falsos para medir qué tan preparados estamos.",
         "Active la verificación en dos pasos en su cuenta corporativa; la guía está en la intranet.",
         "Las claves de los equipos se cambian cada 90 días desde el portal interno, no por correo.",
         "Los computadores se bloquean solos después de 10 minutos sin uso, por política de la empresa."],
        ["Si algo le parece raro, llame a la extensión 1200 antes de responder.", "Gracias por ayudarnos a cuidar la información de todos.",
         "La capacitación virtual es obligatoria y está en el campus de formación.", ""]),
        firma=R.choice(["Seguridad de la Información", "Mesa de Ayuda", "Tecnología", "Gerencia de TI"]))


def l_promo():
    """Publicidad legítima de tiendas: descuentos sin pedir datos ni enlaces raros."""
    t = R.choice(TIENDAS + ["Totto", "Arturo Calle", "Crepes & Waffles", "Cine Colombia", "Juan Valdez"])
    return armar(frases(
        [f"Temporada de vacaciones: {R.randint(2, 6)}0% menos en {R.choice(['maletas', 'morrales', 'gafas de sol', 'vestidos de baño'])}.",
         f"Esta semana tenemos 2x1 en {R.choice(['entradas', 'bebidas', 'postres', 'camisetas'])}.", "Conoce la nueva colección de temporada.",
         f"Acumula puntos con tus compras y redímelos en cualquiera de nuestras tiendas de {R.choice(CIUDADES)}."],
        ["Válido hasta agotar existencias.", "Te esperamos en nuestras tiendas.", "Promoción solo en tiendas físicas.", ""],
        ["Si no quieres recibir más correos, puedes darte de baja aquí.", ""]), firma=t)


def l_trabajo_corto():
    return armar(frases(
        ["Gracias por el apoyo con la presentación de ayer.", f"El café de la oficina se acabó; ¿alguien lo compra el {fecha()}?",
         "Ya subí la versión final del contrato a la carpeta compartida.", f"Mañana llego tarde porque tengo cita médica a las {hora()}.",
         f"El parqueadero estará cerrado el {fecha()} por mantenimiento.", "¿Alguien tiene el cargador de un portátil Lenovo que me preste?"],
        ["Quedo atento.", "Mil gracias.", "Cualquier cosa me escriben al celular.", "Buen trabajo, equipo.", ""]),
        firma=R.choice([persona(), nombre(), "Talento Humano", ""]))


# ---------------------------------------------------------------- más phishing difícil
def p_bono():
    b = R.choice(BANCOS + TIENDAS)
    return armar(frases(
        [f"Fuiste seleccionado para recibir un cashback de {plata()} en {b}.", f"Tienes un reembolso pendiente de {plata()} en {b}.",
         f"Tu tarjeta {b} fue elegida para una mejora de cupo sin costo."],
        [f"Para recibirlo valida tu identidad con tu usuario y clave en {url_falsa(b)}.", "Responde con los 16 dígitos de tu tarjeta y la fecha de vencimiento.",
         f"Reclámalo en {url_falsa(b)} antes de que venza."],
        ["Oferta válida por 24 horas.", "Cupos limitados.", ""]), firma=R.choice([b, ""]))


def p_factura_llamar():
    e = R.choice(EMPRESAS + SERVICIOS)
    return armar(frases(
        [f"Tiene un saldo en mora con {e} por {plata()}.", "Se generó un cobro a su nombre que no ha sido cancelado.",
         f"Su crédito presenta {R.randint(2, 6)} cuotas atrasadas."],
        [f"Para evitar un proceso judicial comuníquese hoy al {R.choice(['3', '6'])}{codigo(9)}.",
         "Abra el archivo adjunto para ver el estado de cuenta.", "Comuníquese con nuestro asesor para un acuerdo de pago inmediato."],
        ["Este es el último aviso.", "Evite cobros jurídicos.", ""]), firma=R.choice(["Cartera", "Departamento jurídico", e]))


def p_datos_discretos():
    return armar(frases(
        [f"Te escribe {persona()}, de la gerencia.", "Te escribo de Contabilidad.", "Soy el nuevo auditor externo."],
        ["Necesito el listado de proveedores con sus números de cuenta, hoy mismo.",
         "Mándame una foto de la tarjeta corporativa por los dos lados para un pago.",
         "Necesito los datos de las tarjetas corporativas para una auditoría."],
        ["Es confidencial, no copies a nadie más.", "No lo comentes todavía con el equipo.", "Es urgente, respóndeme por aquí.", ""]),
        firma=R.choice([persona(), "Gerencia", ""]))


def l_banco_aviso():
    """Avisos reales de bancos: solo informan, no piden datos ni traen enlaces raros."""
    b = R.choice(BANCOS)
    return armar(frases(
        [f"El desembolso de su préstamo por {plata()} ya fue abonado a su cuenta.", f"Se aplicó el pago de su cuota por {plata()}.",
         f"Su nueva tarjeta débito fue enviada a la dirección registrada y llegará en {R.randint(3, 8)} días hábiles.",
         f"El cupo de su tarjeta de crédito aumentó a {plata()} según la solicitud que hizo en la oficina.",
         f"Ya puede descargar el paz y salvo de su crédito desde la app de {b}.",
         f"Su CDT por {plata()} se renovó automáticamente por {R.choice(['90', '180', '360'])} días.",
         f"Registramos el cambio de su número de celular el {fecha()}.",
         f"El {fecha()} de {hora()} a {hora()} la app no estará disponible por mejoras."],
        ["Si usted no hizo esta solicitud, acérquese a cualquier oficina.", "No es necesario que haga nada más.",
         "Recuerde que solo nos comunicamos desde las líneas que aparecen en su tarjeta.", "Gracias por elegirnos.", ""]),
        firma=R.choice([b, f"{b} Colombia", f"Servicio al cliente {b}"]))


def p_conocido():
    """Alguien que dice ser un familiar o amigo, escribe desde otro número o cuenta y pide plata."""
    quien = R.choice(["tu hijo", "tu hija", "tu sobrino", "tu prima", "tu hermano", "tu amiga", "tu compañero de trabajo", ""])
    return armar(frases(
        [f"Soy {quien}, " if quien else "", ""],
        ["te escribo desde otro número porque el mío se mojó.", "este es mi número nuevo, el otro me lo robaron en el bus.",
         "estoy usando el celular de un amigo porque perdí el mío.", "cambié de número, guárdalo así.",
         "me robaron el teléfono y te escribo desde el correo de una amiga.", "se me dañó el celular y estoy en uno prestado."],
        [f"¿Me puedes transferir {plata()} hoy? Te los devuelvo mañana.", f"Necesito que me prestes {plata()} para una urgencia.",
         f"Hazme una consignación de {plata()} a esta cuenta que es de un amigo.", f"¿Me pasas {plata()} por Nequi? Es urgente."],
        ["No llames a nadie que se preocupan.", "Ahorita no puedo hablar, después te explico.", "Es un favor rapidito.", ""]),
        firma=R.choice([nombre(), ""]))


def l_numero_nuevo():
    """Avisos reales de número nuevo: no piden plata."""
    return armar(frases(
        ["Este es mi número nuevo, guárdalo.", "Cambié de operador y ahora tengo otro número.",
         "Me robaron el celular, pero ya recuperé la línea con el mismo número.", "Te escribo por aquí porque se me dañó el celular."],
        ["El otro ya no lo uso.", "Cualquier cosa me escribes a este.", "Avísales a los demás del grupo.", ""],
        ["Un abrazo.", "Nos hablamos.", ""]), firma=nombre())


def p_empleo_cobro():
    """Ofertas de trabajo o negocios que cobran por empezar."""
    return armar(frases(
        [f"Quedó preseleccionado para un cargo con salario de {plata()}.", "Buscamos personas para digitar datos desde la casa.",
         f"Gana {plata()} semanales armando manillas en tu tiempo libre.", "Te ofrecemos un trabajo remoto en una empresa internacional."],
        [f"Para seguir en el proceso debe pagar {plata()} del curso de inducción.", f"El kit de inicio cuesta {plata()} y se paga por adelantado.",
         f"Antes de firmar el contrato hay que cancelar {plata()} de la póliza.", f"Debe consignar {plata()} para la afiliación al sistema."],
        ["Los cupos son limitados.", "Responda hoy para no perder el puesto.", ""]), firma=R.choice(["Recursos Humanos", persona(), ""]))


def l_tramite():
    """Trámites reales con plata o abogados: giros, cesantías, notarías, pensiones."""
    return armar(frases(
        [f"El retiro parcial de sus cesantías por {plata()} fue aprobado y se consignará a su cuenta registrada.",
         "La notaría le informa que la escritura ya está lista para firmar.", f"Su abogado le envía el poder para firmar antes del {fecha()}.",
         f"La devolución de su matrícula por {plata()} se hará a la cuenta que indicó en el formulario.",
         f"El auxilio de transporte de {R.choice(MESES)} se pagará junto con la nómina."],
        ["Si tiene dudas, comuníquese con la oficina en horario de atención.", "No es necesario hacer nada más.", "Lleve su documento original.", ""]),
        firma=R.choice(["Notaría 21", "Fondo de cesantías", "Oficina jurídica", "Tesorería", persona()]))


LEGIT += [l_numero_nuevo, l_tramite, l_banco_aviso, l_plata_familia, l_personal, l_comunidad, l_entidad, l_seguridad_interna, l_promo, l_trabajo_corto]
LEGIT_W += [5, 8, 10, 12, 12, 6, 8, 4, 6, 8]
PHISH += [p_bono, p_factura_llamar, p_datos_discretos, p_conocido, p_empleo_cobro]
PHISH_W += [8, 7, 7, 10, 7]


def generar(n, seed):
    R.seed(seed)
    rows = []
    for _ in range(n):
        f = R.choices(LEGIT, LEGIT_W)[0]
        t = f()
        rows.append((con_enlace(t) if R.random() < 0.3 else t, 0, "Sintetico_es", f.__name__))
        f = R.choices(PHISH, PHISH_W)[0]
        rows.append((f(), 1, "Sintetico_es", f.__name__))
    df = pd.DataFrame(rows, columns=["text", "label", "source", "tipo"]).drop_duplicates("text")
    df["model_text"] = df.text.map(for_model)
    return df[["text", "model_text", "label", "source", "tipo"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500, help="correos por clase para entrenamiento")
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic_es"))
    ap.add_argument("--show", type=int, default=0, help="solo imprimir ejemplos")
    a = ap.parse_args()
    if a.show:
        for _, r in generar(a.show, 7).iterrows():
            print(f"--- [{r.label}] {r.tipo}\n{r.text}\n")
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, va = generar(a.n, 1), generar(max(a.n // 6, 50), 2)
    va = va[~va.text.isin(tr.text)]
    tr.to_csv(out / "train.csv", index=False)
    va.to_csv(out / "val.csv", index=False)
    print(f"Sintéticos: {len(tr)} de entrenamiento, {len(va)} de validación en {out}")
    print(tr.groupby(["label", "tipo"]).size().to_string())


if __name__ == "__main__":
    main()
