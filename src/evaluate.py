import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

def generate_evaluation_graphs(X, y):
    # Create directory for graphs
    os.makedirs("app/static/graphs", exist_ok=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 1. Accuracy vs n_estimators
    # -----------------------------
    estimators = [10, 50, 100, 150, 200]
    accuracies = []

    for n in estimators:
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        accuracies.append(acc)

    plt.figure()
    plt.plot(estimators, accuracies, marker='o')
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs n_estimators")
    plt.grid()

    plt.savefig("app/static/graphs/accuracy_vs_estimators.png")
    plt.close()

    # -----------------------------
    # 2. Accuracy vs max_depth
    # -----------------------------
    depths = [2, 4, 6, 8, 10]
    depth_acc = []

    for d in depths:
        model = RandomForestClassifier(max_depth=d, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        depth_acc.append(acc)

    plt.figure()
    plt.plot(depths, depth_acc, marker='o')
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs max_depth")
    plt.grid()

    plt.savefig("app/static/graphs/accuracy_vs_depth.png")
    plt.close()

    # -----------------------------
    # 3. Confusion Matrix
    # -----------------------------
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title("Confusion Matrix")
    plt.savefig("app/static/graphs/confusion_matrix.png")
    plt.close()

    # -----------------------------
    # 4. ROC Curve
    # -----------------------------
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig("app/static/graphs/roc_curve.png")
