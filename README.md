# ShieldDefence

Proyecto de PTIA (Escuela Colombiana de Ingeniería Julio Garavito). ShieldDefence revisa un
correo y dice si parece phishing. Además le explica al usuario, en español sencillo, por qué
llegó a esa decisión y qué debería hacer.

## Cómo funciona

```
Correo (texto)  ──────────────┐
Pantallazo / PDF ──► OCR ─────┤
                              ▼
                     Preprocesamiento ──► LSTM ──► y (clase), c (confianza)
                              │                          │
                              └──── T (correo original) ─┴──► LLM ──► explicación + recomendación
```

- **LSTM (decide):** BiLSTM entrenada desde cero con Keras. En el servidor corre con NumPy
  (`shield_core/lstm_numpy.py`), con los mismos resultados que Keras, para no cargar TensorFlow.
- **LLM (explica):** Qwen2.5-0.5B-Instruct ajustado con LoRA. Recibe el correo, la decisión de la
  LSTM y las señales encontradas. Nunca decide: si su respuesta contradice a la LSTM o no tiene el
  formato esperado, se descarta y se usa una explicación por plantilla.
- **OCR:** Tesseract en español e inglés, para pantallazos y PDF.
- **Interfaz:** el prototipo tipo chat del Hito 2.

## Versión publicada (todo en el navegador)

La aplicación está en **https://juandagomez12.github.io/ShieldDefence/** (GitHub Pages, carpeta `docs/`).
No tiene servidor: todo corre en el navegador de quien la usa, así que el correo no sale de su computador.

- La BiLSTM se pasó a JavaScript (`docs/js/shield-core.js`) junto con la limpieza del texto, las
  señales y el prompt. `tests/web/parity.mjs` comprueba que da lo mismo que Python.
- El LLM se exporta a ONNX y se cuantiza a 4 bits (`training/export_onnx.py`, unos 440 MB) para correr
  con [Transformers.js](https://huggingface.co/docs/transformers.js). **Por ahora está oculto en la
  página:** con WebGPU genera texto sin sentido, y en CPU tarda cerca de 3 minutos y la versión de 4 bits
  pierde el formato de respuesta. En Colab (PyTorch, GPU T4) sí funciona bien: 100% de respuestas con el
  formato esperado, 100% coherentes con la LSTM y 3.75 s por explicación (paso 9 del notebook).
  Mientras tanto la página muestra la explicación por plantilla. Para probar el LLM: `?llm=1`.
- El OCR usa Tesseract.js y los PDF se leen con PDF.js.
- Los pesos se descargan de Hugging Face:
  [shielddefence-lstm](https://huggingface.co/juandaGomez/shielddefence-lstm) y
  [shielddefence-explicador-onnx](https://huggingface.co/juandaGomez/shielddefence-explicador-onnx).

## Estructura

| Carpeta | Qué tiene |
|---|---|
| `app/` | API con FastAPI (`main.py`), flujo LSTM → LLM (`pipeline.py`) y OCR (`ocr.py`) |
| `shield_core/` | Código compartido: limpieza de texto, señales, explicaciones y la LSTM en NumPy |
| `docs/` | Versión publicada en GitHub Pages (todo en el navegador) |
| `frontend/` | Interfaz tipo chat para la versión con servidor |
| `training/` | Scripts para armar los datos, traducir, entrenar y evaluar |
| `notebooks/` | Notebook de Colab que entrena y exporta todo |
| `models/` | Modelos entrenados y métricas |
| `tests/` | Prueba de que la versión en JavaScript da lo mismo que Python |

## Correrlo en el computador

Versión del navegador: `cd docs && python -m http.server`, y abrir http://localhost:8000.

Versión con servidor (FastAPI):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt          # y tener instalado Tesseract para el OCR
# opcional: export SHIELD_LLM_MODEL=USUARIO/shielddefence-explicador
uvicorn app.main:app --port 7860
```

Luego hay que abrir http://localhost:7860. Sin `SHIELD_LLM_MODEL`, las explicaciones se arman con plantillas.

## Entrenar

Todo está en `notebooks/ShieldDefence_Colab.ipynb`. Hay que abrirlo en Google Colab con GPU T4,
guardar el token de Hugging Face como secreto `HF_TOKEN` y ejecutar las celdas en orden. El notebook:

1. Descarga los datos desde GitHub y arma el dataset balanceado.
2. Traduce la mitad de los correos al español con MarianMT (`Helsinki-NLP/opus-mt-en-es`).
3. Entrena el modelo base (TF-IDF + regresión logística) y la BiLSTM, y exporta la BiLSTM a NumPy.
4. Arma los ejemplos de explicación y ajusta Qwen2.5-0.5B con LoRA.
5. Evalúa el LLM.
6. Exporta la BiLSTM y el LLM para el navegador y los sube a Hugging Face.

## API (versión con servidor)

| Método | Ruta | Entrada |
|---|---|---|
| GET | `/api/estado` | — |
| POST | `/api/analizar` | `{"texto": "..."}` |
| POST | `/api/analizar-archivo` | formulario con `archivo` (PNG, JPG o PDF) |

## Datos

| Clase | Fuente |
|---|---|
| Legítimo | Enron (sin el spam), SpamAssassin (easy_ham y hard_ham), correos legítimos generados con LLM |
| Phishing | Nazario, Nigerian Fraud, correos de phishing generados con LLM |

Copias usadas: [rokibulroni/Phishing-Email-Dataset](https://github.com/rokibulroni/Phishing-Email-Dataset)
(recopilación de Al-Subaiey et al., 2024; CC BY-SA 4.0) y
[cramosme/LLM-Phishing-Detection](https://github.com/cramosme/LLM-Phishing-Detection).
`data/spanish_eval.csv` tiene 30 correos en español escritos por el equipo solo para pruebas. No se
usan para entrenar.

## Sesgo del saludo

En el español traducido, casi solo el phishing empezaba con saludo ("Dear customer" → "Hola"), y la
primera BiLSTM aprendió que "Hola, buenos días" ya era phishing (un correo normal de un profesor salía
con 89%). Ahora `train_lstm.py` agrega saludos y despedidas al azar a los correos de entrenamiento y
validación de las dos clases, y el umbral se elige pidiendo 95% de precisión en vez de 85%.
`lstm_config.json` guarda `falsas_alarmas_por_saludo`: de los correos legítimos de prueba que el modelo
acierta, qué fracción pasa a phishing si se le antepone un saludo.

## Robustez en español y texto disfrazado

El español del dataset original es traducción automática, y el 40% del phishing son estafas
nigerianas en tono personal. El modelo marcaba como phishing correos normales en español:
uno de un profesor, recetas, plata entre familia, avisos reales de bancos. Se hizo esto:

- **Correos sintéticos en español** (`make_synthetic_es.py`): plantillas del día a día y phishing
  difícil, con las dos clases compartiendo saludos, firmas y largo. `check_overlap.py` comprueba que
  ninguno se parezca a los correos de prueba.
- **Palabras desconocidas al entrenar**: el 12% de las palabras se cambia por palabras inventadas,
  porque el modelo había aprendido que una palabra rara era señal de phishing.
- **Quitar disfraces** (`shield_core/deobfuscate.py`): "c l a v e", "c.o.n.t.r.a.s.e.ñ.a", "cl4ve",
  caracteres invisibles, letras cirílicas iguales a las latinas y letras de adorno. El texto limpio
  es el que ven la LSTM y las reglas, y el disfraz se marca como la señal "Texto disfrazado".
- **Combinador** (`shield_core/combine.py`): una regresión logística decide con la BiLSTM y las
  señales de las reglas. Las señales de riesgo solo pueden subir la probabilidad.

- **Reglas de casos frecuentes**: alguien conocido que escribe desde otro número o cuenta y pide plata,
  trabajos o plata fácil que cobran por empezar, y cobros pequeños para liberar un paquete.
- **Combinaciones decisivas**: combinaciones de señales que casi nunca tiene un correo legítimo
  (por ejemplo, conocido desde otro número + pide plata) marcan el correo aunque la LSTM dude. Solo
  se usan las que activan a lo sumo el 0.2% de los legítimos de validación.
- **Recomendaciones según el caso**: llamar al familiar al número de siempre, entrar a la app
  oficial del banco o llamar al número de la tarjeta, revisar el pedido en la app de la tienda,
  consultar la página .gov.co de la entidad, no pagar por ofertas de trabajo.

Cada prueba ciega se escribió antes de ver resultados y se usó una sola vez, con la versión que
había en ese momento:

| Prueba | Versión | Resultado |
|---|---|---|
| Prueba ciega 1 (40 correos) | LSTM + señales | 90%: 0 falsas alarmas, 16 de 20 phishing |
| Prueba ciega 2 (40 correos) | + reglas de suplantación | 85%: 4 falsas alarmas, 18 de 20 phishing |
| **Prueba ciega 3 (30 correos)** | **+ combinaciones decisivas (actual)** | **100%: 0 falsas alarmas, 15 de 15 phishing** |
| Español escrito a mano (125, usada para ajustar) | actual | 95.2% |
| Phishing en español disfrazado | actual | 98% |
| Prueba general (2584 correos) | actual | 97.5% (LSTM sola: 98.5%) |

El costo: en la prueba general el combinador marca más correos legítimos generados con IA (que
hablan mucho de cuentas y contraseñas): 46 falsas alarmas de 1291, contra 19 de la LSTM sola.

## Limitaciones

- Casi todos los datos son en inglés y de otros años. El español depende de la traducción automática.
- La LSTM solo mira el texto: no revisa adjuntos, encabezados técnicos ni qué hay detrás de un enlace.
- El LLM todavía no funciona bien en el navegador (ver arriba), así que la página publicada explica
  con plantillas. El LLM ajustado sí funciona en la versión con servidor (FastAPI).
