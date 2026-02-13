from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import hashlib
import uuid
import time
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input


app = FastAPI(title="Plastic Verification Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model(
    "model/plastic_model.keras",
    custom_objects={"preprocess_input": preprocess_input}
)


class_names = ["OTHERS", "PC", "PE", "PET", "PP", "PS"]

def preprocess_image(image):
    image = image.resize((260, 260))  # CHANGE if different
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def home():
    return {"status": "Backend running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    request_id = str(uuid.uuid4())

    image_hash = hashlib.sha256(contents).hexdigest()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img = preprocess_image(image)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return JSONResponse({
        "request_id": request_id,
        "plastic_type": class_names[class_index],
        "confidence": confidence,
        "image_hash": image_hash,
        "timestamp": time.time(),
        "verification_status": "verified"
    })
