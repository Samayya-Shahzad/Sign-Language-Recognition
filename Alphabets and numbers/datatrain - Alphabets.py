import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import sys
import codecs
import matplotlib.pyplot as plt

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Define constants
IMAGE_SIZE = 100  # Size of images (100x100)
NUM_CLASSES = len(['A','B','C','D','dot','E','F','G','H','I','J','K','L','M','N','num','O','P','Q','R','S','space','T','U','V','W','words','X','Y','Z'])  # Number of gesture categories

# Load the preprocessed data
with open("X_al.pickle", "rb") as f:
    X = pickle.load(f)
with open("y_al.pickle", "rb") as f:
    y = pickle.load(f)

# Normalize image data (scaling pixel values to the range 0-1)
X = X / 255.0

# One-hot encode the labels
y = to_categorical(y, NUM_CLASSES)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Define the model
model = Sequential()
model.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))  # Input layer for color images (3 channels)

model.add(Conv2D(16, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Increase dropout rate
model.add(Dense(NUM_CLASSES, activation='softmax'))  # Output layer with softmax activation

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(datagen.flow(X, y, batch_size=32, subset='training'),
                    validation_data=datagen.flow(X, y, batch_size=32, subset='validation'),
                    epochs=50,  # Increase epochs since early stopping will handle overfitting
                    callbacks=[early_stopping])



# Plot training and validation accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the trained model
model.save('asl_alphabet_model.h5')

print("Model training complete. Model saved as 'asl_model.h5'.")