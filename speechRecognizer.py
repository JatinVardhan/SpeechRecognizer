import numpy as np

def compute_cost(w, X, y, C):
    hinge_loss = np.maximum(0, 1 - y * (X.dot(w)))
    cost = 0.5 * np.dot(w, w) + C * np.sum(hinge_loss)
    return cost

def compute_gradient(w, X, y, C):
    grad = w.copy()
    for i in range(len(y)):
        if y[i] * np.dot(X[i], w) < 1:  # Check if margin is less than 1
            grad -= C * y[i] * X[i]
    return grad

def train_svm(X, y, C=1.0, learning_rate=0.01, epochs=500):
    w = np.zeros(X.shape[1])
    for epoch in range(epochs):
        grad = compute_gradient(w, X, y, C)
        w -= learning_rate * grad  # Update weights
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {compute_cost(w, X, y, C)}")
    return w

def predict(X, w):
    return np.sign(X.dot(w))

X = np.array([[2, 3], [1, 1], [2, 1], [3, 2], [1, 3], [2, 2]])
y = np.array([1, 1, -1, -1, 1, -1])

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # Normalize data

weights = train_svm(X, y)  # Train the model

predictions = predict(X, weights)  # Make predictions
print("Predictions:", predictions)
