import numpy as np

# Load the extracted features and labels
features = np.load("features.npy")
labels = np.load("labels.npy")

# Print the shape and an example feature and label
print(f"Features shape: {features.shape}")
print(f"Example feature vector: {features[0]}")
print(f"Example label: {labels[0]}")
