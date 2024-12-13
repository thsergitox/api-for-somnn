from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import pickle

app = FastAPI()
som_model = None
M = None

# Cargar el modelo SOM entrenado y la matriz M
try:
    with open("som_mnist.pkl", "rb") as tf:
        som_model = pickle.load(tf)
    with open("M_matrix.pkl", "rb") as mf:
        M = pickle.load(mf)
except Exception as e:
    raise Exception(f"Error al cargar el modelo o la matriz M: {str(e)}")


def preprocess_image(image):
    global som_model, M
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
    global som_model, M
    try:
        # Preprocesar la imagen para hacerla coincidir con la entrada del modelo
        # a un array 784
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
