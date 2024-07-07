# Implementation of a CNN model for CIFAR-10 dataset with validation split

import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import plot_model
from keras.datasets import cifar10
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

# Definition of class labels
labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# Load the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Data preprocessing
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert labels to one-hot encoding
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=42)

# Define the model using Functional API
input_shape = (32, 32, 3)
inputs = Input(shape=input_shape)
x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, batch_size=1000, epochs=3, verbose='auto', validation_data=(x_val, y_val))

# Save training history plot
df = pd.DataFrame(history.history)
ax = df.plot()
fig = ax.get_figure()
fig.savefig('/mnt/data/history_plot.png')

# Model evaluation
loss, accuracy = model.evaluate(x_test, y_test, verbose='auto')
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Model visualization
model.summary()
plot_model(model, to_file='/mnt/data/model_plot.png', show_shapes=True, show_layer_names=True)

# Export the model
model.save("/mnt/data/my_model_cifar10.keras")

# Displaying misclassified examples
predictions = np.argmax(model.predict(x_test), axis=1)
y_test_flat = np.argmax(y_test, axis=1)
incorrect_indices = np.nonzero(predictions != y_test_flat)[0]
for i in range(5):
    idx = incorrect_indices[i]
    print("Misclassified example number", i+1)
    plt.imshow(x_test[idx])
    plt.xlabel(f"True label: {labels[y_test_flat[idx]]}, Predicted label: {labels[predictions[idx]]}")
    plt.show()

# Creating confusion matrix
cm = confusion_matrix(y_test_flat, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# Creating the plot
fig, ax = plt.subplots(figsize=(10, 10))
disp = disp.plot(xticks_rotation='vertical', ax=ax, cmap='summer')

# Save the plot to a file
plt.savefig('/mnt/data/confusion_matrix.png')
plt.show()
