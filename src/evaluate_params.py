"""
Module for evaluating models with different hyperparameters
Generates performance metrics for parameter tuning
"""
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot

import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.models.random_forest import RandomForest
from src.utils.data_preprocessing import load_data
from src.utils.metrics import accuracy, precision, recall, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "heart_kaggle.csv"
MODEL_PATH = ROOT_DIR / "models" / "random_forest_model.pkl"

# Store test data globally (loaded once)
_TEST_DATA = None


def get_test_data():
    """Load test data from saved model checkpoint"""
    global _TEST_DATA
    if _TEST_DATA is None:
        if MODEL_PATH.exists():
            with MODEL_PATH.open("rb") as f:
                checkpoint = pickle.load(f)
            _TEST_DATA = {
                "X_test": checkpoint["X_test"],
                "y_test": checkpoint["y_test"]
            }
        else:
            # Fallback: load data and split
            X, y = load_data(DATA_PATH)
            np.random.seed(42)
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            split = int(0.8 * len(X))
            _TEST_DATA = {
                "X_test": X[split:],
                "y_test": y[split:]
            }
    return _TEST_DATA


def train_and_evaluate_model(n_trees, max_depth, min_samples_split=2):
    """
    Train a Random Forest model with given parameters and return evaluation metrics
    """
    # Get test data
    test_data = get_test_data()
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    
    # Reload training data for training
    X, y = load_data(DATA_PATH)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    
    # Train model
    model = RandomForest(
        n_trees=int(n_trees),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        n_features=int(np.sqrt(X.shape[1]))
    )
    
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = _get_proba(model, X_test)
    
    # Calculate metrics
    acc = accuracy(y_test, y_pred)
    prec = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Generate confusion matrix image
    cm_image = _generate_confusion_matrix_image(y_test, y_pred)
    
    # ROC-AUC
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = 0.0
    
    return {
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "specificity": float(specificity),
            "sensitivity": float(sensitivity),
            "roc_auc": float(roc_auc)
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "confusion_matrix_image": cm_image,
        "roc_curve": {
            "fpr": fpr.tolist() if 'fpr' in locals() else [],
            "tpr": tpr.tolist() if 'tpr' in locals() else []
        }
    }


def _get_proba(model, X):
    """Get probability predictions for ROC curve"""
    predictions = np.array([tree.predict(X) for tree in model.trees])
    predictions = np.swapaxes(predictions, 0, 1)
    
    # Average predictions as probability
    return np.mean(predictions, axis=1)


def _generate_confusion_matrix_image(y_test, y_pred):
    """
    Generate confusion matrix image as base64 PNG using seaborn
    """
    try:
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        print(f"Error generating confusion matrix image: {e}")
        raise


def evaluate_parameter_range(param_name, param_range, base_params=None):
    """
    Evaluate model performance across a range of parameter values
    
    Args:
        param_name: 'n_trees', 'max_depth', or 'min_samples_split'
        param_range: list of values to test
        base_params: dict with default parameter values
    
    Returns:
        dict with results for each parameter value
    """
    if base_params is None:
        base_params = {"n_trees": 100, "max_depth": 5, "min_samples_split": 2}
    
    results = []
    
    for param_value in param_range:
        params = base_params.copy()
        params[param_name] = param_value
        
        try:
            evaluation = train_and_evaluate_model(
                params["n_trees"],
                params["max_depth"],
                params["min_samples_split"]
            )
            
            results.append({
                param_name: param_value,
                "accuracy": evaluation["metrics"]["accuracy"],
                "precision": evaluation["metrics"]["precision"],
                "recall": evaluation["metrics"]["recall"],
                "f1_score": evaluation["metrics"]["f1_score"],
                "roc_auc": evaluation["metrics"]["roc_auc"]
            })
        except Exception as e:
            print(f"Error evaluating {param_name}={param_value}: {e}")
            results.append({
                param_name: param_value,
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "roc_auc": 0
            })
    
    return {
        "param_name": param_name,
        "results": results
    }


def get_default_best_params():
    """Return the best known parameters"""
    return {
        "n_trees": 100,
        "max_depth": 5,
        "min_samples_split": 2
    }
