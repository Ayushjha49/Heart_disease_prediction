import numpy as np

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for i in range(len(y_true)):
        true_idx = class_to_index[y_true[i]]
        pred_idx = class_to_index[y_pred[i]]
        matrix[true_idx][pred_idx] += 1

    return matrix


def precision(y_true, y_pred, average='weighted'):
    """Calculate precision score."""
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    
    precisions = []
    for i in range(len(classes)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        if tp + fp == 0:
            precisions.append(0)
        else:
            precisions.append(tp / (tp + fp))
    
    if average == 'weighted':
        class_counts = [np.sum(y_true == cls) for cls in classes]
        return np.average(precisions, weights=class_counts)
    return np.mean(precisions)


def recall(y_true, y_pred, average='weighted'):
    """Calculate recall score."""
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    
    recalls = []
    for i in range(len(classes)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        if tp + fn == 0:
            recalls.append(0)
        else:
            recalls.append(tp / (tp + fn))
    
    if average == 'weighted':
        class_counts = [np.sum(y_true == cls) for cls in classes]
        return np.average(recalls, weights=class_counts)
    return np.mean(recalls)


def f1_score(y_true, y_pred, average='weighted'):
    """Calculate F1 score."""
    prec = precision(y_true, y_pred, average)
    rec = recall(y_true, y_pred, average)
    if prec + rec == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)