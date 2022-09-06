import numpy as np
import json
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from job_prepr_model.ml_logic.registry import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.state.model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    params = data['params']
    arr = np.fromstring(data['arr'],dtype=float).reshape(48,48,1)
    pred = app.state.model.predict(arr)
    return pred

@app.get('/hello')
def test():
    return 'Hello'
