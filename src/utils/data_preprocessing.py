import pandas as pd

# All columns are already numeric in heart_kaggle.csv — no encoding needed
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]
TARGET_COLUMN = "target"

# float columns — everything else is int
FLOAT_COLUMNS = {"oldpeak"}


def load_data(path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    return X, y


def encode_input(form_data):
    """Convert raw form strings to numeric values for model input."""
    values = []
    for col in FEATURE_COLUMNS:
        raw_val = form_data[col]
        if col in FLOAT_COLUMNS:
            values.append(float(raw_val))
        else:
            values.append(int(float(raw_val)))  # int(float()) handles "120.0" safely
    return values