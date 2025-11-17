from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import io
import os
import numpy as np
import tensorflow as tf

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DEFAULT_MODEL_PATHS = [os.path.join(PROJECT_DIR, "models", "mnist_cnn.keras")]
_model = None
_model_path = None

def _resolve_model_path():
    for p in DEFAULT_MODEL_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Модель не найдена.")

def get_model():
    global _model, _model_path
    if _model is None:
        path = _resolve_model_path()
        _model_path = path
        _model = tf.keras.models.load_model(path)
    return _model

def preprocess_image(img: Image.Image):
    img = img.convert("L")
    img = ImageOps.fit(img, (28, 28), method=Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(img: Image.Image):
    model = get_model()
    x = preprocess_image(img)
    probs = model.predict(x, verbose=0)[0]
    cls = int(np.argmax(probs))
    return cls, probs.tolist()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("L")
    try:
        pred, probs = predict_image(img)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"prediction": int(pred), "probabilities": probs}