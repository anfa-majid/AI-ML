import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization


# Set your data directory
data_dir = './dataset'

# Hyperparameters
batch_size = 32
input_shape = (64, 64, 3)
epochs = 20
patience = 10


leaky_relu = LeakyReLU(alpha=0.1)

#model = Sequential()

#model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
#model.add(leaky_relu)  # Replace ReLU with Leaky ReLU
#model.add(BatchNormalization())  # Add Batch Normalization
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

# Define the CNN architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(leaky_relu)  # Replace ReLU with Leaky ReLU
model.add(BatchNormalization())  # Add Batch Normalization
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(leaky_relu)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, kernel_regularizer=l2(0.001)))
model.add(leaky_relu)
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))  # Replace '3' with the actual number of classes in your dataset


# Use a smaller initial learning rate
from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare the data generators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical')

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min', restore_best_weights=True)

# Add a learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

# Train the model
history = model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=[early_stopping, reduce_lr])

# Save the model
model.save('asl_cnn_model.h5')

# Evaluate the model on the test set
scores = model.evaluate(test_generator)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

import matplotlib.pyplot as plt

# Plot the training and validation loss over time
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy over time
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

