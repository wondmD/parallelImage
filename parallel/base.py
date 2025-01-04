
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# PARTS TO BE DONE SEQUENTIALLY
# 1. Model Initialization
strategy = tf.distribute.MirroredStrategy()  # Utilize multiple GPUs
with strategy.scope():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5, seed=42),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

# 2. Data Loading and Preprocessing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Data Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(x_train)

# PARTS TO BE DONE IN PARALLEL
# Data Splitting into Chunks for Batch Processing
batch_size = 256
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Train the model with distributed strategy
model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=50, validation_data=(x_test, y_test))

# PARTS TO BE DONE SEQUENTIALLY
# Save the model
model.save('cifar10_model.h5')

# Epoch Evaluation: Evaluate model on validation dataset
val_loss, val_accuracy = model.evaluate(x_test, y_test)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# TEST PART
# Load the model
loaded_model = tf.keras.models.load_model('cifar10_model.h5')

# Test the model with a random image from the test set
index = np.random.randint(0, x_test.shape[0])  # Pick a random index from the test set
test_image = x_test[index]  # Get the corresponding test image
test_label = np.argmax(y_test[index])  # Get the true label for the image

# Show the test image
plt.imshow(test_image)
plt.title(f"True Label: {test_label}")
plt.show()

# Expand dimensions to match the input shape (batch size, height, width, channels)
test_image = np.expand_dims(test_image, axis=0)

# Make a prediction
prediction = loaded_model.predict(test_image)
predicted_label = np.argmax(prediction)

print(f"Predicted Label: {predicted_label}")
