import os

# Redirect Keras cache to D: drive to avoid "No space left on device" error on C:
cache_path = 'D:/DHWANI/ENGINEERING/VI/IPCV/keras_cache'
os.environ['KERAS_HOME'] = cache_path
if not os.path.exists(cache_path):
    os.makedirs(cache_path)
# --------------------
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np



# 1. Load CIFAR-10 Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# Class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# 2. Build Model
model = models.Sequential([
    # Input shape is (32, 32, 3) for RGB images
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 3. Train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Note: 10-20 epochs is better for CIFAR-10, but keeping 3 for speed
history = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

# A. Plot Accuracy & Loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()

# B. Show Predictions for first 5 test images
plt.figure(figsize=(12, 3))
for i in range(5):
    img = x_test[i]
    # Predict the class
    pred_array = model.predict(img.reshape(1, 32, 32, 3), verbose=0)
    pred = np.argmax(pred_array)
    
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    # y_test[i][0] fetches the integer label for the i-th image
    plt.title(f"T: {class_names[y_test[i][0]]}\nP: {class_names[pred]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
