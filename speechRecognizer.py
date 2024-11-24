import numpy as np
<<<<<<< HEAD

def compute_cost(w, X, y, C):
    hinge_loss = np.maximum(0, 1 - y * (X.dot(w)))
    cost = 0.5 * np.dot(w, w) + C * np.sum(hinge_loss)
=======
import matplotlib.pyplot as plt

def compute_cost(w, X, y, C):
    h = np.maximum(0, 1 - y * (X.dot(w)))
    cost = 0.5 * np.dot(w, w) + C * np.sum(h)
>>>>>>> f8a78aa (first commit)
    return cost

def compute_gradient(w, X, y, C):
    grad = w.copy()
    for i in range(len(y)):
<<<<<<< HEAD
        if y[i] * np.dot(X[i], w) < 1:  # Check if margin is less than 1
=======
        if y[i] * np.dot(X[i], w) < 1:  # Checking if margin is less than 1
>>>>>>> f8a78aa (first commit)
            grad -= C * y[i] * X[i]
    return grad

def train_svm(X, y, C=1.0, learning_rate=0.01, epochs=500):
    w = np.zeros(X.shape[1])
    for epoch in range(epochs):
        grad = compute_gradient(w, X, y, C)
<<<<<<< HEAD
        w -= learning_rate * grad  # Update weights
        if epoch % 100 == 0:
=======
        w -= learning_rate * grad  # Updating weights/parameter betas
        if epoch % 10 == 0:
>>>>>>> f8a78aa (first commit)
            print(f"Epoch {epoch}, Cost: {compute_cost(w, X, y, C)}")
    return w

def predict(X, w):
    return np.sign(X.dot(w))

<<<<<<< HEAD
X = np.array([[2, 3], [1, 1], [2, 1], [3, 2], [1, 3], [2, 2]])
y = np.array([1, 1, -1, -1, 1, -1])

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # Normalize data

weights = train_svm(X, y)  # Train the model

predictions = predict(X, weights)  # Make predictions
print("Predictions:", predictions)
=======
def visualize_svm(X, y, w):
    # Create a mesh grid for plotting the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict the values for the grid
    Z = np.dot(grid, w)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and margins
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'green', 'blue'], alpha=0.3)
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'green', 'blue'], linestyles=['--', '-', '--'])
 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=100, edgecolors='k', label='Data points')
    
    plt.title("SVM with Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc='upper left')
    plt.show()


X = np.array([[2, 3], [1, 1], [2, 1], [3, 2], [1, 3], [2, 2]])
y = np.array([1, 1, -1, -1, 1, -1])

# X = np.array([
#     [2, 3], [1, 1], [2, 1], [3, 2], [1, 3], [2, 2],
#     [3, 3], [4, 1], [3, 1], [2, 4], [1, 4], [4, 3], 
#     [3, 5], [4, 5], [5, 4], [5, 1], [6, 2], [4, 2],
#     [5, 3], [6, 3], [6, 1], [7, 3], [7, 2]
# ])

# y = np.array([1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1])

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  

weights = train_svm(X, y)  # Training the model

predictions = predict(X, weights)  # Making predictions
print("Predictions:", predictions)

# Call the visualization function
visualize_svm(X, y, weights)
>>>>>>> f8a78aa (first commit)
