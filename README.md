# 💰 Loan Approval Prediction System
A Machine Learning-based system to predict loan approval using FastAPI backend and Streamlit frontend, with Docker support.
## 📌 Overview
This project predicts whether a loan application will be approved or rejected using Machine Learning. It uses applicant details like income, credit history, and loan amount to make accurate predictions.

---

## 🚀 Features
- Data preprocessing & cleaning  
- Feature engineering  
- Model training (Logistic Regression, Random Forest, Gradient Boosting)  
- Model comparison & evaluation  
- Visualization of results  
- FastAPI-based prediction API  
- Interactive Streamlit frontend  
- Docker deployment  

---

## 🛠️ Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- FastAPI  
- Streamlit  
- Docker  

---

## ⚙️ How to Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt



### Run Backend
python -m uvicorn app.main:app --reload

### Run Frontend
streamlit run frontend.py

---

## 🐳 Docker Setup

Build image:
docker build -t loan-api .

Run container:
docker run -p 8000:8000 loan-api

---

## 🔌 API Endpoint

POST /predict  
Input: applicant details  
Output: Approved / Rejected

---

## 🚀 Demo
- FastAPI Docs: http://127.0.0.1:8000/docs  
- Frontend UI: http://localhost:8501

## 👤 Author
**Kruthika S**  
🔗 GitHub: https://github.com/kruthikasrinivasa

## 📊 Results
The following visualizations show model performance and evaluation metrics.

## 📊 Results

### Model Comparison
![Model Comparison](artifacts/plots/model_comparison.png)

### ROC Curve
![ROC Curve](artifacts/plots/roc_curve_comparison.png)

### Confusion Matrix (Best Model)
![Confusion Matrix](artifacts/plots/best_model_confusion_matrix.png)

### Feature Importance
![Feature Importance](artifacts/plots/feature_importance.png)