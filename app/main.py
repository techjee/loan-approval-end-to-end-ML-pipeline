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
    
    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # Feature engineering (same as training)
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["IncomeLoanRatio"] = df["TotalIncome"] / df["LoanAmount"]

    # Simple input validation (safe feature)
    if df["LoanAmount"][0] <= 0:
        return {"error": "Invalid Loan Amount"}

    # Prediction
    prediction = model.predict(df)[0]

    # Small enhancement: decision message
    if prediction == 1:
        message = "Eligible for loan based on current details"
    else:
        message = "Not eligible. Improve income or credit history"

    return {
        "prediction": int(prediction),
        "result": "Approved" if prediction == 1 else "Rejected",
        "message": message
    }