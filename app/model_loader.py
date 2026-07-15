import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "fruad_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scalers.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "features_names.pkl"))
    return model, scaler, feature_names