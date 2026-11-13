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
- El LLM se exporta a ONNX y se cuantiza a 4 bits (`training/export_onnx.py`, unos 400 MB). Corre con
  [Transformers.js](https://huggingface.co/docs/transformers.js), con WebGPU si el navegador lo tiene.
  Es opcional: se activa con el botón "Activar explicación con IA" y el navegador lo guarda después
  de la primera descarga. Mientras tanto se muestra la explicación por plantilla.
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
`data/spanish_eval.csv` tiene 24 correos en español escritos por el equipo solo para pruebas. No se
usan para entrenar.

## Limitaciones

- Casi todos los datos son en inglés y de otros años. El español depende de la traducción automática.
- La LSTM solo mira el texto: no revisa adjuntos, encabezados técnicos ni qué hay detrás de un enlace.
- El LLM en el navegador pesa unos 400 MB y en un computador sin WebGPU puede tardar bastante
  en escribir cada explicación. Por eso es opcional.
