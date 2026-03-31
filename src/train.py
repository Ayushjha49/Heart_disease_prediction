from src.models.random_forest import RandomForest
from src.utils.data_preprocessing import load_data
from src.utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns



ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "heart_kaggle.csv"
MODEL_PATH = ROOT_DIR / "models" / "random_forest_model.pkl"
REPORTS_PATH = ROOT_DIR / "reports"


def train():

    X, y = load_data(DATA_PATH)


    print("Class distribution:", np.bincount(y))


    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]


    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]


    model = RandomForest(
        n_trees=100,                        
        max_depth=5,
        min_samples_split=5,                       
        n_features=int(np.sqrt(X.shape[1]))
    )


    model.fit(X_train, y_train)
    print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")

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


def save_metrics_charts(model, X_test, y_test, reports_path=REPORTS_PATH):
    """Generate and save precision, recall, f1-score, and confusion matrix charts."""
    reports_path.mkdir(parents=True, exist_ok=True)
    

    y_pred = model.predict(X_test)
    

    acc = accuracy(y_test, y_pred)
    prec = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("TEST SET METRICS")
    print("="*50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*50 + "\n")
    

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_values = [acc, prec, rec, f1]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Metrics (Test Set)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(reports_path / 'metrics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Metrics chart saved to: {reports_path / 'metrics.png'}")
    plt.close()
    
    # 2. Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(reports_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to: {reports_path / 'confusion_matrix.png'}")
    plt.close()


if __name__ == "__main__":
    trained_model, X_test, y_test = train()
    save_model(trained_model, X_test, y_test)
    save_metrics_charts(trained_model, X_test, y_test)
    print(f"✅ Model saved to: {MODEL_PATH}")
