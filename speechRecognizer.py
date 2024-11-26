import numpy as np
import matplotlib.pyplot as plt

def polynomial_features(X):
    # Transforming input features into polynomial features.
    X_poly = []
    for x1, x2 in X:
        X_poly.append([x1, x2, x1**2, x2**2, x1 * x2])
    return np.array(X_poly)

def compute_cost(w, X, y, C):
    h = np.maximum(0, 1 - y * (X.dot(w)))
    cost = 0.5 * np.dot(w, w) + C * np.sum(h)
    return cost

def compute_gradient(w, X, y, C):
    grad = w.copy()
    for i in range(len(y)):
        if y[i] * np.dot(X[i], w) < 1:  # Checking if margin is less than 1
            grad -= C * y[i] * X[i]
    return grad

def train_svm(X, y, C=1.0, learning_rate=0.01, epochs=500):
    w = np.zeros(X.shape[1])
    for epoch in range(epochs):
        grad = compute_gradient(w, X, y, C)
        w -= learning_rate * grad  # Updating weights/parameter betas
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Cost: {compute_cost(w, X, y, C)}")
    return w

def predict(X, w):
    return np.sign(X.dot(w))

def calculate_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    accuracy = correct / len(y_true)
    return accuracy


def visualize_svm(X, y, w):
    # Create a mesh grid for plotting the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Transform the grid points to polynomial features
    grid_poly = polynomial_features(grid)
    Z = np.dot(grid_poly, w)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and margins
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'green', 'blue'], alpha=0.3)
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'green', 'blue'], linestyles=['--', '-', '--'])
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=100, edgecolors='k', label='Data points')
    plt.title("SVM with Polynomial Kernel")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc='upper left')
    plt.show()

# Non-linearly separable data
# X = np.array([
#     [1, 2], [2, 1], [1, -1], [2, -2], [-1, 1], [-2, 1],
#     [-1, -1], [-2, -2], [3, 3], [3, -3], [-3, 3], [-3, -3]
# ])
# y = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1])
X = np.array([
    [2, 3], [1, 1], [2, 1], [3, 2], [1, 3], [2, 2],
    [3, 3], [4, 1], [3, 1], [2, 4], [1, 4], [4, 3], 
    [3, 5], [4, 5], [5, 4], [5, 1], [6, 2], [4, 2],
    [5, 3], [6, 3], [6, 1], [7, 3], [7, 2]
])

y = np.array([1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1])
# Normalize data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Transform features to polynomial space
X_poly = polynomial_features(X)

# Train SVM
weights = train_svm(X_poly, y)

# Making predictions
X_poly_test = polynomial_features(X)
predictions = predict(X_poly_test, weights)
print("Predictions:", predictions)

# Calculating accuracy
accuracy = calculate_accuracy(y, predictions)
print("Accuracy:", accuracy)

# Visualize the decision boundary
visualize_svm(X, y, weights)

