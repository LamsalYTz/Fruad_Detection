# from fastapi import FastAPI, Security, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from app.auth import verify_security_key
# from app.model_loader import load_artifacts
# from app.preprocessor import preprocess_data
# from fastapi import UploadFile, File
# import pandas as pd

# app = FastAPI(
#     title="🔍 Fraud Detection API",
#     description="""
# ## AI-Powered Credit Card Fraud Detection API

# Detect fraudulent transactions in real time using Machine Learning.

# ### How it works
# Submit transaction data via **POST /detect** and get back:
# - Fraud prediction (Yes/No)
# - Confidence score
# - Risk level (High/Medium/Low)
# - Recommended action

# ### Authentication
# Pass your API key as `X-API-Key` in the request header.
#     """,
#     version="1.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# model, scaler, feature_names = load_artifacts()

# @app.get("/", tags=["General"])
# def root():
#     return {
#         "name": "Fraud Detection API",
#         "version": "1.0.0",
#         "docs": "/docs"
#     }

# @app.get("/health", tags=["General"])
# def health():
#     return {
#         "status": "ok",
#         "model": "loaded",
#         "features": len(feature_names)
#     }

# @app.post("/detect", tags=["Detection"])
# async def detect_fraud(
#     file: UploadFile = File(...),
#     api_key: str = Security(verify_security_key)
# ):
#     try:
#         # Read uploaded CSV
#         df = pd.read_csv(file.file)

#         # Store predictions
#         predictions = []

#         for _, row in df.iterrows():

#             data = row.to_dict()

#             processed = preprocess_data(
#                 data,
#                 scaler,
#                 feature_names
#             )

#             prediction = model.predict(processed)[0]
#             probability = float(model.predict_proba(processed)[0][1])

#             is_fraud = prediction == 1
#             confidence = round(
#                 probability if is_fraud else 1 - probability,
#                 3
#             )

#             if probability >= 0.7:
#                 risk = "High"
#                 recommendation = "Block transaction immediately and alert the cardholder."
#             elif probability >= 0.4:
#                 risk = "Medium"
#                 recommendation = "Flag for manual review before processing."
#             else:
#                 risk = "Low"
#                 recommendation = "Transaction appears legitimate."

#             predictions.append({
#                 "Time": data["Time"],
#                 "Amount": data["Amount"],
#                 "Prediction": "Fraud" if is_fraud else "Legitimate",
#                 "Confidence": confidence,
#                 "Risk": risk,
#                 "Recommendation": recommendation
#             })

#         return {
#             "total_transactions": len(predictions),
#             "results": predictions
#         }

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))


import joblib
from fastapi import FastAPI

app = FastAPI()

# Load model once
model = joblib.load("models/fruad_model.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API"}

@app.get("/predict")
def predict(amount: float):

    prediction = model.predict([[amount]])[0]

    if prediction == 1:
        result = "Fraud"
    else:
        result = "Legitimate"

    return {
        "amount": amount,
        "prediction": result
    }