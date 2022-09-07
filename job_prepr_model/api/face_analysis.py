import numpy as np
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from job_prepr_model.ml_logic.registry import load_model

#Instanciate API
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#Load model on API launch
app.state.model = load_model()

@app.post("/predict")
async def predict(array: Request):
    """Return a prediction from a webrtc frame"""
    data = await array.json()
    arr = np.array(eval(data))
    pred = app.state.model.predict(arr.reshape(1,48,48,1))
    return pred.tolist()
