import pandas as pd

def preprocess_data(data : dict, scaler, feature_names: list):
    df = pd.DataFrame([data])
    df = df[feature_names]
    scaled = scaler.transform(df)
    return scaled
