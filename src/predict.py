import numpy as np
import pickle
from pathlib import Path

_MODEL = None
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "random_forest_model.pkl"


def _get_model():
    global _MODEL
    if _MODEL is None:
        if MODEL_PATH.exists():
            # Load from saved checkpoint
            with MODEL_PATH.open("rb") as f:
                checkpoint = pickle.load(f)
            _MODEL = checkpoint["model"]
        else:
            # No saved model — train, save, then use
            from src.train import train, save_model
            print("⚠️  No saved model found. Training now...")
            _MODEL, X_test, y_test = train()
            save_model(_MODEL, X_test, y_test)
            print(f"✅ Model saved to: {MODEL_PATH}")
    return _MODEL


def predict(input_data):
    model = _get_model()
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return "Heart Disease" if prediction[0] == 1 else "No Heart Disease"


if __name__ == "__main__":
    sample = [40, 1, 1, 140, 289, 0, 0, 172, 0, 0.0, 2,3,0]
    print(predict(sample))