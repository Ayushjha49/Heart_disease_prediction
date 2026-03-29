from src.models.random_forest import RandomForest
from src.utils.data_preprocessing import load_data
import numpy as np
import pickle
from pathlib import Path


# Root directory setup
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "heart_kaggle.csv"
MODEL_PATH = ROOT_DIR / "models" / "random_forest_model.pkl"


def train():
    # Load dataset
    X, y = load_data(DATA_PATH)

    # 🔍 Check class distribution (useful for debugging + viva)
    print("Class distribution:", np.bincount(y))

    # 🔥 STEP 1: Shuffle data (VERY IMPORTANT — fixed seed for reproducibility)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # 🔥 STEP 2: Split dataset (80% train, 20% test)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 🔥 STEP 3: Initialize Random Forest
    model = RandomForest(
        n_trees=100,                         # more trees → better performance
        max_depth=5,                        # deeper trees → better learning
        n_features=int(np.sqrt(X.shape[1]))  # standard RF rule
    )

    # Train model
    model.fit(X_train, y_train)
    print(model.score(X_train, y_train))


    return model, X_test, y_test


def save_model(model, X_test, y_test, model_path=MODEL_PATH):
    """Save model + test data together so evaluate.py can use the exact same split."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
    }
    with model_path.open("wb") as f:
        pickle.dump(checkpoint, f)


if __name__ == "__main__":
    trained_model, X_test, y_test = train()
    save_model(trained_model, X_test, y_test)
    print(f"Model saved to: {MODEL_PATH}")
