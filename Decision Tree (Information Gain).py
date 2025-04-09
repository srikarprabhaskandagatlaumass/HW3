import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import Counter

class DecisionNode:
    def __init__(self, feature=None, threshold=None, descendants=None, category=None):
        self.feature = feature
        self.threshold = threshold
        self.descendants = descendants or {}
        self.category = category

class RandomForest:
    def __init__(self, n_trees, k_folds, max_depth):
        self.n_trees = n_trees
        self.k_folds = k_folds
        self.max_depth = max_depth
        self.trees = []
        self.feature_importances = None

    def entropy(self, sample):
        label_counts = Counter(row[-1] for row in sample)
        total_samples = len(sample)
        return -sum((count / total_samples) * np.log2(count / total_samples)
                   for count in label_counts.values() if count > 0)

    def is_numeric(self, value):
        return isinstance(value, (int, float, np.number))

    def information_gain(self, sample, available_features, all_features):
        def feature_entropy(feature):
            feature_idx = all_features.index(feature)
            feature_values = [row[feature_idx] for row in sample]

            if self.is_numeric(feature_values[0]):
                if feature_values:
                    threshold = np.nanmean(feature_values)
                else:
                    threshold = 0

                left_subset = [row for row in sample if row[feature_idx] <= threshold]
                right_subset = [row for row in sample if row[feature_idx] > threshold]

                if not left_subset or not right_subset:
                    return 0

                weighted_entropy = (len(left_subset) / len(sample)) * self.entropy(left_subset) + \
                                   (len(right_subset) / len(sample)) * self.entropy(right_subset)
                return self.entropy(sample) - weighted_entropy
            else:
                feature_values = set(feature_values)
                weighted_entropy = sum(
                    len(subset) / len(sample) * self.entropy(subset)
                    for value in feature_values
                    if (subset := [row for row in sample if row[feature_idx] == value]))

                return self.entropy(sample) - weighted_entropy

        return max(available_features, key=feature_entropy)

    def create_bootstrap_sample(self, dataset):
        n_samples = len(dataset)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return [dataset[i] for i in indices]

    def construct_tree(self, sample, all_features):
        def _construct_tree(sample, available_features, current_depth=0):
            if not sample:
                return DecisionNode(category=None)

            labels = [row[-1] for row in sample]
            if len(set(labels)) == 1 or current_depth >= self.max_depth:
                most_common = Counter(labels).most_common(1)
                return DecisionNode(category=most_common[0][0] if most_common else None)

            m = int(np.sqrt(len(all_features)))
            selected_features = random.sample(available_features, min(m, len(available_features)))

            if not selected_features:
                most_common = Counter(labels).most_common(1)
                return DecisionNode(category=most_common[0][0] if most_common else None)

            best_feature = self.information_gain(sample, selected_features, all_features)
            root = DecisionNode(feature=best_feature)
            feature_idx = all_features.index(best_feature)
            feature_values = [row[feature_idx] for row in sample]

            if self.is_numeric(feature_values[0]):
                if feature_values:
                    threshold = np.nanmean(feature_values)
                else:
                    threshold = 0
                root.threshold = threshold
                left_subset = [row for row in sample if row[feature_idx] <= threshold]
                right_subset = [row for row in sample if row[feature_idx] > threshold]
                remaining_features = [f for f in available_features if f != best_feature]
                root.descendants['left'] = _construct_tree(left_subset, remaining_features, current_depth + 1)
                root.descendants['right'] = _construct_tree(right_subset, remaining_features, current_depth + 1)
            else:
                for value in set(feature_values):
                    subset = [row for row in sample if row[feature_idx] == value]
                    remaining_features = [f for f in available_features if f != best_feature]
                    root.descendants[value] = _construct_tree(subset, remaining_features, current_depth + 1)

            return root

        return _construct_tree(sample, all_features, 0)

    def classify_sample(self, tree, sample, features):
        if tree.category is not None:
            return tree.category

        if tree.feature is None:
            return None

        feature_value = sample[features.index(tree.feature)]

        if tree.threshold is not None:
            direction = 'left' if feature_value <= tree.threshold else 'right'
            if direction in tree.descendants:
                return self.classify_sample(tree.descendants[direction], sample, features)
        elif feature_value in tree.descendants:
            return self.classify_sample(tree.descendants[feature_value], sample, features)

        descendant_categories = [
            self.classify_sample(descendant, sample, features)
            for descendant in tree.descendants.values()
        ]

        valid_categories = [cat for cat in descendant_categories if cat is not None]
        if valid_categories:
            return Counter(valid_categories).most_common(1)[0][0]
        else:
            return None

    def majority_vote(self, predictions):
        valid_predictions = [p for p in predictions if p is not None]
        if valid_predictions:
            return Counter(valid_predictions).most_common(1)[0][0]
        else:
            return None

    def stratified_kfold_split(self, X, y, k):
        data = np.column_stack((X, y))
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}
        folds = [[] for _ in range(k)]

        for cls in unique_classes:
            indices = class_indices[cls]
            random.shuffle(indices)
            samples_per_fold = len(indices) // k
            remainder = len(indices) % k
            start = 0

            for i in range(k):
                end = start + samples_per_fold + (1 if i < remainder else 0)
                fold_indices = indices[start:end]
                folds[i].extend(data[fold_indices])
                start = end

        for i in range(k):
            test_data = np.array(folds[i])
            train_data = np.concatenate([np.array(fold) for j, fold in enumerate(folds) if j != i])
            X_train, y_train = train_data[:, :-1], train_data[:, -1]
            X_test, y_test = test_data[:, :-1], test_data[:, -1]
            yield X_train, X_test, y_train, y_test

    def fit(self, X, y, features):
        dataset = [list(X[i]) + [y[i]] for i in range(len(X))]
        self.trees = []

        for _ in range(self.n_trees):
            bootstrap_sample = self.create_bootstrap_sample(dataset)
            tree = self.construct_tree(bootstrap_sample, features)
            self.trees.append(tree)

        self._calculate_feature_importances(dataset, features)

    def _calculate_feature_importances(self, dataset, features):
        self.feature_importances = {f: 0 for f in features}
        total_samples = len(dataset)

        for tree in self.trees:
            self._update_feature_importance(tree, dataset, features, total_samples)

        total = sum(self.feature_importances.values())
        if total > 0:
            for f in self.feature_importances:
                self.feature_importances[f] /= total

    def _update_feature_importance(self, node, dataset, features, total_samples, impurity_reduction=0.0):
        if len(dataset) > 0:
            labels = [row[-1] for row in dataset]
            original_variance = np.var(labels)
        else:
            original_variance = 0.0

        if node.feature is not None:
            feature_idx = features.index(node.feature)
            if node.threshold is not None:
                left_subset = [row for row in dataset if row[feature_idx] <= node.threshold]
                right_subset = [row for row in dataset if row[feature_idx] > node.threshold]

                left_variance = np.var([row[-1] for row in left_subset]) if left_subset else 0.0
                right_variance = np.var([row[-1] for row in right_subset]) if right_subset else 0.0

                weighted_variance = (len(left_subset) / len(dataset)) * left_variance + \
                                    (len(right_subset) / len(dataset)) * right_variance
                impurity_reduction = original_variance - weighted_variance

                if 'left' in node.descendants and left_subset:
                    self._update_feature_importance(node.descendants['left'], left_subset, features, total_samples,
                                                    impurity_reduction)
                if 'right' in node.descendants and right_subset:
                    self._update_feature_importance(node.descendants['right'], right_subset, features,
                                                    total_samples, impurity_reduction)
            else:
                for value, child_node in node.descendants.items():
                    subset = [row for row in dataset if row[feature_idx] == value]
                    if subset:
                        self._update_feature_importance(child_node, subset, features, total_samples,
                                                        impurity_reduction)

            self.feature_importances[node.feature] += impurity_reduction

    def predict(self, X, features):
        predictions = []
        for sample in X:
            tree_predictions = [self.classify_sample(tree, sample, features) for tree in self.trees]
            predictions.append(self.majority_vote(tree_predictions))
        return predictions

    def calculate_accuracy(self, y_true, y_pred):
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true)

    def calculate_precision(self, y_true, y_pred, positive_class):
        true_positives = sum(
            1 for true, pred in zip(y_true, y_pred) if true == positive_class and pred == positive_class)
        predicted_positives = sum(1 for pred in y_pred if pred == positive_class)
        epsilon = 1e-9
        return true_positives / (predicted_positives + epsilon)

    def calculate_recall(self, y_true, y_pred, positive_class):
        true_positives = sum(
            1 for true, pred in zip(y_true, y_pred) if true == positive_class and pred == positive_class)
        actual_positives = sum(1 for true in y_true if true == positive_class)
        epsilon = 1e-9
        return true_positives / (actual_positives + epsilon)

    def calculate_f1_score(self, y_true, y_pred, positive_class):
        precision = self.calculate_precision(y_true, y_pred, positive_class)
        recall = self.calculate_recall(y_true, y_pred, positive_class)
        epsilon = 1e-9
        return 2 * (precision * recall) / (precision + recall + epsilon)

    def evaluate(self, X, y, features):
        predictions = self.predict(X, features)
        classes = list(set(y))
        precisions = []
        recalls = []
        f1_scores = []
        epsilon = 1e-9

        for cls in classes:
            precisions.append(self.calculate_precision(y, predictions, cls))
            recalls.append(self.calculate_recall(y, predictions, cls))
            f1_scores.append(self.calculate_f1_score(y, predictions, cls))

        class_counts = Counter(y)
        total_samples = len(y)

        weighted_precision = sum(
            precisions[i] * class_counts[cls] for i, cls in enumerate(classes)) / total_samples
        weighted_recall = sum(recalls[i] * class_counts[cls] for i, cls in enumerate(classes)) / total_samples
        weighted_f1 = sum(f1_scores[i] * class_counts[cls] for i, cls in enumerate(classes)) / total_samples

        accuracy = self.calculate_accuracy(y, predictions)

        return accuracy, weighted_precision, weighted_recall, weighted_f1

def load_dataset(filename):
    df = pd.read_csv(filename)

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].map({val: i for i, val in enumerate(df[col].unique())})
            except:
                print(f"Could not convert column {col}")
                continue

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    X = df.drop('label', axis=1).values
    y = df['label'].values
    features = df.drop('label', axis=1).columns.tolist()

    return X, y, features


if __name__ == '__main__':
    n_trees_list = [1, 5, 10, 20, 30, 40, 50]
    k_folds = 5
    max_depth = 20

    X, y, features = load_dataset('titanic.csv')

    results = {
        'accuracy': {ntree: [] for ntree in n_trees_list},
        'precision': {ntree: [] for ntree in n_trees_list},
        'recall': {ntree: [] for ntree in n_trees_list},
        'f1': {ntree: [] for ntree in n_trees_list}
    }

    splitter = RandomForest(n_trees=n_trees_list[0], k_folds=k_folds,
                            max_depth=max_depth)

    print(f"\nStarting {k_folds}-fold cross-validation for different ntree values...")
    for ntree in n_trees_list:
        print(f"\nProcessing ntree = {ntree}")
        kfold_generator = splitter.stratified_kfold_split(X, y, k_folds)
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []

        for fold, (X_train, X_test, y_train, y_test) in enumerate(kfold_generator):
            print(f" Fold {fold + 1}/{k_folds}", end='\r')
            fold_rf = RandomForest(n_trees=ntree, k_folds=k_folds,
                                    max_depth=max_depth)
            fold_rf.fit(X_train, y_train, features)
            accuracy, precision, recall, f1 = fold_rf.evaluate(X_test, y_test, features)
            fold_accuracies.append(accuracy)
            fold_precisions.append(precision)
            fold_recalls.append(recall)
            fold_f1s.append(f1)

        results['accuracy'][ntree] = np.mean(fold_accuracies)
        results['precision'][ntree] = np.mean(fold_precisions)
        results['recall'][ntree] = np.mean(fold_recalls)
        results['f1'][ntree] = np.mean(fold_f1s)

        print(f" ntree = {ntree} completed - Accuracy: {results['accuracy'][ntree]:.4f}, "
              f"Precision: {results['precision'][ntree]:.4f}, Recall: {results['recall'][ntree]:.4f}, "
              f"F1 Score: {results['f1'][ntree]:.4f}")

    plt.figure(figsize=(12, 8))
    plt.plot(n_trees_list, [results['accuracy'][ntree] for ntree in n_trees_list], label='Accuracy', marker='o')
    plt.plot(n_trees_list, [results['precision'][ntree] for ntree in n_trees_list], label='Precision', marker='o')
    plt.plot(n_trees_list, [results['recall'][ntree] for ntree in n_trees_list], label='Recall', marker='o')
    plt.plot(n_trees_list, [results['f1'][ntree] for ntree in n_trees_list], label='F1 Score', marker='o')

    plt.xlabel('Number of Trees (ntree)')
    plt.ylabel('Score')
    plt.title('Random Forest Performance vs. Number of Trees')
    plt.legend()
    plt.grid(True)
    plt.show()