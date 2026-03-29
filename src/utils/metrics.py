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