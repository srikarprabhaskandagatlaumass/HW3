# Importing the required libraries
import numpy as np
import pandas as pd
from collections import Counter
import random
import matplotlib.pyplot as plt

# Hyper-parameters
n_trees = 50
k_folds = 5
random_state = 42

# Decision Tree Node
class DecisionNode:
    def __init__(self, feature=None, threshold=None, descendants=None, category=None):
        self.feature = feature
        self.threshold = threshold
        self.descendants = descendants or {}
        self.category = category

# Random Forest Implementation
class RandomForest:
    def __init__(self, n_trees=10, k_folds=5, random_state=None, max_depth=5):
        self.n_trees = n_trees
        self.k_folds = k_folds
        self.random_state = random_state
        self.max_depth = max_depth  # New parameter for maximal depth
        self.trees = []
        self.feature_importances = None
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
    
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
                threshold = np.mean(feature_values)
                left_subset = [row for row in sample if row[feature_idx] <= threshold]
                right_subset = [row for row in sample if row[feature_idx] > threshold]
                
                if not left_subset or not right_subset:
                    return 0
                
                weighted_entropy = (len(left_subset)/len(sample)) * self.entropy(left_subset) + \
                                  (len(right_subset)/len(sample)) * self.entropy(right_subset)
                return self.entropy(sample) - weighted_entropy
            else:
                feature_values = set(feature_values)
                weighted_entropy = sum(
                    len(subset)/len(sample) * self.entropy(subset)
                    for value in feature_values
                    if (subset := [row for row in sample if row[feature_idx] == value])
                )
                return self.entropy(sample) - weighted_entropy
        
        return max(available_features, key=feature_entropy)
    
    def create_bootstrap_sample(self, dataset):
        n_samples = len(dataset)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return [dataset[i] for i in indices]
    
    def construct_tree(self, sample, all_features):
        def _construct_tree(sample, available_features, current_depth=0):  # Add current_depth parameter
            labels = [row[-1] for row in sample]
            
            # Stop if max depth reached
            if current_depth >= self.max_depth:
                return DecisionNode(category=Counter(labels).most_common(1)[0][0])
            
            # Existing stopping conditions
            if len(set(labels)) == 1:
                return DecisionNode(category=labels[0])
            
            if not available_features:
                return DecisionNode(category=Counter(labels).most_common(1)[0][0])
            
            # Random feature selection
            m = int(np.sqrt(len(all_features)))
            selected_features = random.sample(available_features, min(m, len(available_features)))
            
            best_feature = self.information_gain(sample, selected_features, all_features)
            root = DecisionNode(feature=best_feature)
            
            feature_idx = all_features.index(best_feature)
            feature_values = [row[feature_idx] for row in sample]
            
            if self.is_numeric(feature_values[0]):
                threshold = np.mean(feature_values)
                root.threshold = threshold
                
                left_subset = [row for row in sample if row[feature_idx] <= threshold]
                right_subset = [row for row in sample if row[feature_idx] > threshold]
                
                remaining_features = [f for f in available_features if f != best_feature]
                
                if left_subset:
                    root.descendants['left'] = _construct_tree(left_subset, remaining_features, current_depth+1)
                else:
                    root.descendants['left'] = DecisionNode(category=Counter(labels).most_common(1)[0][0])
                    
                if right_subset:
                    root.descendants['right'] = _construct_tree(right_subset, remaining_features, current_depth+1)
                else:
                    root.descendants['right'] = DecisionNode(category=Counter(labels).most_common(1)[0][0])
            else:
                for value in set(feature_values):
                    subset = [row for row in sample if row[feature_idx] == value]
                    if subset:
                        remaining_features = [f for f in available_features if f != best_feature]
                        root.descendants[value] = _construct_tree(subset, remaining_features, current_depth+1)
                    else:
                        root.descendants[value] = DecisionNode(category=Counter(labels).most_common(1)[0][0])
            
            return root
        
        return _construct_tree(sample, all_features.copy(), 0)  # Start with depth 0
    
    def classify_sample(self, tree, sample, features):
        if tree.category is not None:
            return tree.category
        
        feature_value = sample[features.index(tree.feature)]
        
        if tree.threshold is not None:
            if feature_value <= tree.threshold:
                direction = 'left'
            else:
                direction = 'right'
            
            if direction in tree.descendants:
                return self.classify_sample(tree.descendants[direction], sample, features)
            else:
                if tree.descendants:
                    return self.classify_sample(next(iter(tree.descendants.values())), sample, features)
                else:
                    return None
        else:
            if feature_value not in tree.descendants:
                descendant_categories = [d.category for d in tree.descendants.values() if d.category]
                return Counter(descendant_categories).most_common(1)[0][0] if descendant_categories else \
                       self.classify_sample(next(iter(tree.descendants.values())), sample, features)
            
            return self.classify_sample(tree.descendants[feature_value], sample, features)
    
    def majority_vote(self, predictions):
        return Counter(predictions).most_common(1)[0][0]
    
    def stratified_kfold_split(self, X, y, k):
        # Combine X and y for easier handling
        data = np.column_stack((X, y))
        
        # Get unique classes and their counts
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}
        
        # Initialize folds
        folds = [[] for _ in range(k)]
        
        # Distribute samples from each class to folds
        for cls in unique_classes:
            indices = class_indices[cls]
            np.random.shuffle(indices)
            
            # Calculate how many samples go to each fold
            samples_per_fold = len(indices) // k
            remainder = len(indices) % k
            
            start = 0
            for i in range(k):
                end = start + samples_per_fold + (1 if i < remainder else 0)
                fold_indices = indices[start:end]
                folds[i].extend(data[fold_indices])
                start = end
        
        # Convert each fold to numpy array and split into train/test
        for i in range(k):
            test_data = np.array(folds[i])
            train_data = np.concatenate([np.array(fold) for j, fold in enumerate(folds) if j != i])
            
            X_train, y_train = train_data[:, :-1], train_data[:, -1]
            X_test, y_test = test_data[:, :-1], test_data[:, -1]
            
            yield X_train, X_test, y_train, y_test
    
    def fit(self, X, y, features):
        # Convert to list of lists for easier handling
        dataset = [list(X[i]) + [y[i]] for i in range(len(X))]
        
        # Build the forest
        self.trees = []
        for _ in range(self.n_trees):
            bootstrap_sample = self.create_bootstrap_sample(dataset)
            tree = self.construct_tree(bootstrap_sample, features)
            self.trees.append(tree)
        
        # Calculate feature importances (optional)
        self._calculate_feature_importances(dataset, features)
    
    def _calculate_feature_importances(self, dataset, features):
        self.feature_importances = {f: 0 for f in features}
        
        for tree in self.trees:
            self._update_feature_importance(tree, dataset, features)
        
        # Normalize importances
        total = sum(self.feature_importances.values())
        if total > 0:
            for f in self.feature_importances:
                self.feature_importances[f] /= total
    
    def _update_feature_importance(self, node, dataset, features, depth=1):
        if node.feature is not None:
            self.feature_importances[node.feature] += depth * len(dataset)
            
            feature_idx = features.index(node.feature)
            if node.threshold is not None:
                left_subset = [row for row in dataset if row[feature_idx] <= node.threshold]
                right_subset = [row for row in dataset if row[feature_idx] > node.threshold]
                
                if 'left' in node.descendants and left_subset:
                    self._update_feature_importance(node.descendants['left'], left_subset, features, depth+1)
                if 'right' in node.descendants and right_subset:
                    self._update_feature_importance(node.descendants['right'], right_subset, features, depth+1)
            else:
                for value, child_node in node.descendants.items():
                    subset = [row for row in dataset if row[feature_idx] == value]
                    if subset:
                        self._update_feature_importance(child_node, subset, features, depth+1)
    
    def predict(self, X, features):
        predictions = []
        for sample in X:
            tree_predictions = [self.classify_sample(tree, sample, features) for tree in self.trees]
            predictions.append(self.majority_vote(tree_predictions))
        return predictions
    
    def evaluate(self, X, y, features):
        predictions = self.predict(X, features)
        correct = sum(1 for true, pred in zip(y, predictions) if true == pred)
        return correct / len(y)

# Load and preprocess data
dataset = pd.read_csv('wdbc.csv', header=None)
header = dataset.iloc[0].tolist()
full_data = dataset.iloc[1:].values

# Convert numerical columns to float (assuming some columns might be numerical)
for i in range(full_data.shape[1]):
    try:
        full_data[:, i] = full_data[:, i].astype(float)
    except ValueError:
        pass  # Leave as string if conversion fails

X = full_data[:, :-1]
y = full_data[:, -1]
features = header[:-1]

# Initialize lists to store results
fold_accuracies = []
feature_importances = []

# Stratified k-fold cross-validation
rf = RandomForest(n_trees=n_trees, k_folds=k_folds, random_state=random_state)
kfold = rf.stratified_kfold_split(X, y, k_folds)

for fold, (X_train, X_test, y_train, y_test) in enumerate(kfold):
    print(f"\nTraining fold {fold + 1}/{k_folds}...")
    
    # Train random forest
    rf.fit(X_train, y_train, features)
    
    # Evaluate on test set
    accuracy = rf.evaluate(X_test, y_test, features)
    fold_accuracies.append(accuracy)
    feature_importances.append(rf.feature_importances)
    
    print(f"Fold {fold + 1} accuracy: {accuracy:.4f}")

# Calculate average performance
mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)

print("\nCross-validation results:")
print(f"Mean accuracy: {mean_accuracy:.4f}")
print(f"Standard deviation: {std_accuracy:.4f}")

# Calculate average feature importances
avg_feature_importance = {}
for f in features:
    avg_feature_importance[f] = np.mean([imp[f] for imp in feature_importances])

# Plot feature importances
plt.figure(figsize=(10, 6))
sorted_features = sorted(avg_feature_importance.items(), key=lambda x: x[1], reverse=True)
features_sorted = [f[0] for f in sorted_features]
importances_sorted = [f[1] for f in sorted_features]
plt.barh(features_sorted, importances_sorted)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.show()

# Plot accuracy distribution
plt.figure(figsize=(10, 6))
plt.hist(fold_accuracies, bins=20, edgecolor='black')
plt.axvline(mean_accuracy, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_accuracy:.4f}')
plt.axvline(mean_accuracy + std_accuracy, color='green', linestyle='dotted', linewidth=1, label=f'Mean + Std: {mean_accuracy + std_accuracy:.4f}')
plt.axvline(mean_accuracy - std_accuracy, color='green', linestyle='dotted', linewidth=1, label=f'Mean - Std: {mean_accuracy - std_accuracy:.4f}')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Random Forest Accuracy Distribution Across Folds')
plt.legend()
plt.show()