import os
import csv
import librosa
import numpy as np

# Function to read the TSV file
def read_tsv(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            data.append(row)
    return data

# Function to extract MFCC features
def extract_mfcc(file_path, n_mfcc=13):
    try:
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=None)  # Use the original sampling rate
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        # Return the mean of each MFCC coefficient over time
        return np.mean(mfcc, axis=1)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process the dataset
def process_dataset(clips_dir, tsv_path):
    data = read_tsv(tsv_path)
    features = []
    labels = []

    # Iterate through TSV entries
    for entry in data:
        mp3_filename = entry["path"]
        text_label = entry["sentence"]

        # Construct the full path to the MP3 file
        mp3_path = os.path.join(clips_dir, mp3_filename)

        # Check if the file exists
        if not os.path.exists(mp3_path):
            print(f"File not found: {mp3_path}")
            continue

        # Extract MFCC features
        mfcc_features = extract_mfcc(mp3_path)
        if mfcc_features is not None:
            features.append(mfcc_features)
            labels.append(text_label)

    return np.array(features), labels

# Main script
clips_dir = "C:\\Users\\dell\\Desktop\\Ml_project\\cv-corpus-18.0-delta-2024-06-14\\en\\clips"
tsv_path = "C:\\Users\\dell\\Desktop\\Ml_project\\cv-corpus-18.0-delta-2024-06-14\\en\\validated.tsv"

# Extract features and labels
features, labels = process_dataset(clips_dir, tsv_path)

# Save the features for future use
np.save("features.npy", features)
np.save("labels.npy", labels)

print("Feature extraction complete.")
print(f"Number of samples: {len(features)}")
