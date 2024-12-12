from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import pickle

app = FastAPI()

# Cargar el modelo SOM entrenado y la matriz M
try:
    with open("som_mnist.pkl", "rb") as tf:
        som_model = pickle.load(tf)
    with open("M_matrix.pkl", "rb") as mf:
        M = pickle.load(mf)
except Exception as e:
    print(f"Error al cargar el modelo o la matriz M: {str(e)}")


def preprocess_image(image):
    # Convertir a escala de grises si es necesario
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Redimensionar a 28x28
    image = cv2.resize(image, (28, 28))

    # Normalizar valores de píxeles
    image = image / 255.0

    # Remodelar para coincidir con la entrada del modelo (784 dimensiones)
    image = image.reshape(784)

    return image


def predict_digit(image):
    try:
        # Preprocesar la imagen
        processed_image = preprocess_image(image)

        # Obtener la neurona ganadora
        winner = som_model.winner(processed_image)

        # Obtener la clase predicha desde la posición de la neurona ganadora
        prediction = M[winner]  # M debe ser cargado con el modelo

        return int(prediction)
    except Exception as e:
        raise Exception(f"Error de predicción: {str(e)}")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer y convertir imagen
        image = Image.open(BytesIO(await file.read()))
        image = np.array(image)

        # Obtener predicción
        prediction = predict_digit(image)

        return {
            "prediction": prediction,
            "message": f"El dígito predicho es: {prediction}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Opcional: Agregar un endpoint simple de verificación de salud
@app.get("/")
async def root():
    return {"message": "La API del Clasificador de Dígitos MNIST está funcionando"}
