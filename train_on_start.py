import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

MODEL_DIR = "models"

def train_and_save():
    print("🔄 Training fraud detection model...")

    # Find the CSV file
    data_files = os.listdir("data")
    csv_file = [f for f in data_files if f.endswith('.csv')][0]
    df = pd.read_csv(f"data/{csv_file}")
    print(f"✅ Loaded: {csv_file} — Shape: {df.shape}")

    X = df.drop('Class', axis=1)
    y = df['Class']

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/fruad_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scalers.pkl")
    joblib.dump(feature_names, f"{MODEL_DIR}/features_names.pkl")
    print("✅ All artifacts saved!")

if __name__ == "__main__":
    train_and_save()