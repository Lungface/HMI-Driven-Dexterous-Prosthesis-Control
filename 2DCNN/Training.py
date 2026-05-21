import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to train, validation, and test folders
train_dir = r"C:\Users\laibi\Desktop\FYP\1.1.0FULLSECTIONS\Session_1_Split\DATA_FOREARM\train"
val_dir = r"C:\Users\laibi\Desktop\FYP\1.1.0FULLSECTIONS\Session_1_Split\DATA_FOREARM\val"
test_dir = r"C:\Users\laibi\Desktop\FYP\1.1.0FULLSECTIONS\Session_1_Split\DATA_FOREARM\test"

# Define image dimensions and batch size
img_height, img_width = 128, 128  # Adjust based on your data
batch_size = 32

# Create ImageDataGenerators for loading data
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

model = models.Sequential()

# CNN layers for spatial feature extraction
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten and reshape for LSTM input
model.add(layers.Flatten())
model.add(layers.Reshape((1, -1)))  # Reshape to (time_steps, features)

# LSTM layers for temporal modeling
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.LSTM(64))

# Fully connected layers for classification
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(train_generator.num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.summary()

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
model.save("cnn_lstm_model.h5")