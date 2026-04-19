from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("artifacts/best_model_pipeline.pkl")


@app.get("/")
def home():
    return {"message": "Loan Prediction API Running"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # Feature engineering: same as training
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["IncomeLoanRatio"] = df["TotalIncome"] / df["LoanAmount"]

    prediction = model.predict(df)[0]

    return {
        "prediction": int(prediction),
        "result": "Approved" if prediction == 1 else "Rejected"
    }