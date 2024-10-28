# K-Nearest Neighbors (KNN) From Scratch

This notebook demonstrates a complete, step-by-step implementation of the K-Nearest Neighbors (KNN) algorithm from scratch. It covers key concepts, code implementation, and model evaluation.

## Objectives
- Understand the basics of the KNN algorithm.
- Implement the KNN algorithm without using specialized libraries.
- Evaluate the model's performance on test data.
- Visualize the data and results to understand KNN behavior.

## Steps

### 1. Import Libraries

The required libraries for numerical operations and visualization are imported first.

```python
import numpy as np
import matplotlib.pyplot as plt
```

### 2. Define Distance Function (e.g., Euclidean)

A function to calculate the distance between two points, usually the Euclidean distance, is implemented.

```python
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
```

### 3. Implement KNN Function from Scratch

The core KNN function is implemented manually, which includes:
- Finding the nearest neighbors to a new sample.
- Using the `k` parameter to choose the number of neighbors.
- Assigning the most common class among neighbors as the class of the new sample.

```python
from collections import Counter

def knn(X_train, y_train, X_test, k=3):
    y_pred = []
    for x_test in X_test:
        distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        y_pred.append(most_common[0][0])
    return np.array(y_pred)
```

### 4. Data Preparation

The data is split into training and test sets. For simplicity, a small dataset such as `Iris` may be used.

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. Model Evaluation

The model's accuracy on the test set is computed.

```python
y_pred = knn(X_train, y_train, X_test, k=3)
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"Accuracy: {accuracy}")
```

### 6. Data and Results Visualization

Charts and plots are generated to illustrate data points and the KNN classification boundaries. This helps visualize the effects of `k` values and data distribution.

```python
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=20)
plt.title("Data Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```
