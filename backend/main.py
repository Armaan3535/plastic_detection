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
import os
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from blockchain import Blockchain
import asyncio
from pydantic import BaseModel


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

PORT = os.getenv("PORT", "8000")
blockchain_file = f"blockchain_{PORT}.json"
blockchain = Blockchain(blockchain_file=blockchain_file)

class_names = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]

class NodeRequest(BaseModel):
    address: str

def preprocess_image(image):
    image = image.resize((224, 224)) 
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
    #print(prediction)

    record = {
        "request_id": request_id,
        "plastic_type": class_names[class_index],
        "confidence": confidence,
        "image_hash": image_hash,
        "timestamp": time.time(),
        "verification_status": "verified"
    }

    blockchain.add_block_from_data(record)

    return JSONResponse(record)

@app.get("/chain")
def get_chain():
    blockchain.load_chain()
    return [b.to_dict() for b in blockchain.chain]

@app.get("/validate")
def validate_chain():
    is_valid = blockchain.is_chain_valid()
    return {"blockchain_valid": is_valid}

@app.post("/register-node")
def register_node(node: NodeRequest):
    blockchain.register_node(node.address)
    return {
        "message": "Node added successfully",
        "total_nodes": list(blockchain.nodes)
    }

@app.get("/resolve")
def resolve():
    replaced = blockchain.resolve_conflicts()

    if replaced:
        return {"message": "Chain was replaced"}
    else:
        return {"message": "Chain is authoritative"}

@app.on_event("startup")
async def start_monitor():
    asyncio.create_task(integrity_monitor())

async def integrity_monitor():
    while True:
        await asyncio.sleep(3)  # check every 3 seconds
        if not blockchain.is_chain_valid():
            print("Live tampering detected! Attempting recovery...")
            blockchain.resolve_conflicts(force=True)
