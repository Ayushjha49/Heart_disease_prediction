import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=5, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)

        best_feature, best_thresh = self._best_split(X, y, feature_idxs)
        if best_feature is None:
            return self.Node(value=self._most_common_label(y))

        left_idxs = X[:, best_feature] <= best_thresh
        right_idxs = X[:, best_feature] > best_thresh

        if not np.any(left_idxs) or not np.any(right_idxs):
            return self.Node(value=self._most_common_label(y))

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth+1)

        return self.Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feature_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feature in feature_idxs:
            thresholds = np.unique(X[:, feature])

            for thresh in thresholds:
                gain = self._information_gain(y, X[:, feature], thresh)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = thresh

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        left_idxs = X_column <= threshold
        right_idxs = X_column > threshold

        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])

        e_l = self._entropy(y[left_idxs])
        e_r = self._entropy(y[right_idxs])

        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r

        return parent_entropy - child_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p*np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)