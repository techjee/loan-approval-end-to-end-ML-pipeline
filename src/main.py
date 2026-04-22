from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# load your model (change path if needed)
model = joblib.load("../artifacts/best_model_pipeline.pkl")

@app.get("/")
def home():
    return {"message": "Loan Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    features = np.array(list(data.values())).reshape(1, -1)
    prediction = model.predict(features)[0]
    return {"result": int(prediction)}