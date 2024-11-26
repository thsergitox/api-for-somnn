from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import cv2
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()

def detect_objects(image):
    # Cargar los clasificadores
    human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar humanos
    humans = human_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

    # Determinar el resultado

    if len(humans) > 0:
        return "Hay una cara"
    else:
        return "No hay una cara"

@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
        image = np.array(image)
        prediction = detect_objects(image)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
