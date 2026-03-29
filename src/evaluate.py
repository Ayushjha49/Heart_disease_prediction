import pickle
from pathlib import Path

from src.utils.metrics import accuracy, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "random_forest_model.pkl"


def evaluate():
    if not MODEL_PATH.exists():
        print("❌ No saved model found. Run 'python -m src.train' first.")
        return

    # Load model + test data saved together during training
    with MODEL_PATH.open("rb") as f:
        checkpoint = pickle.load(f)

    model  = checkpoint["model"]
    X_test = checkpoint["X_test"]
    y_test = checkpoint["y_test"]

    print(f"✅ Loaded model from: {MODEL_PATH}")
    print(f"   Test set size: {len(y_test)} samples")

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    evaluate()