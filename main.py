import joblib
import pandas as pd

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

df = pd.read_csv("data/creditcard.csv").sample(100)  # small sample
X = df.drop("Class", axis=1)
X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)
df["Predicted"] = preds
print(df.head())
