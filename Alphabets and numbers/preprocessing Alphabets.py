from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
import pickle

# Define dataset directory and categories
DATADIR = "Data/Alphabets"
CATEGORIES = ['A','B','C','D','dot','E','F','G','H','I','J','K','L','M','N','num','O','P','Q','R','S','space','T','U','V','W','words','X','Y','Z']
IMG_SIZE = 100  # Resize images to 100x100

# Function to create training data
def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # Path to each category
        class_num = CATEGORIES.index(category)  # Assign numerical label
        for img in os.listdir(path):  # Iterate over each image in the category
            try:
                img_array = cv2.imread(os.path.join(path, img))  # Read image in color (default mode)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # Resize to 100x100
                training_data.append([new_array, class_num])  # Add to training data
            except Exception as e:
                pass
    return training_data

# Create and shuffle the training data
training_data = create_training_data()
np.random.shuffle(training_data)  # Shuffle data for better training

# Split features (X) and labels (y)
X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

# Convert to numpy arrays
X = np.array(X)  # Color images already have 3 channels (no need to reshape)
y = np.array(y)

# Save data using pickle
with open("X_al.pickle", "wb") as f:
    pickle.dump(X, f)
with open("y_al.pickle", "wb") as f:
    pickle.dump(y, f)

print("Data preprocessing complete. Features and labels saved as X_color.pickle and y_color.pickle.")