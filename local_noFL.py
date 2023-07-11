# Standard Libraries
import os
import time

# Third Party Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, image as img
from memory_profiler import memory_usage
from pathlib import Path
from progressbar import progressbar
from skimage.transform import resize
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPool2D,
)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf



# random seed generator
np.random.seed(3)
tf.random.set_seed(7)

# # Measure the execution time
# start_time = time.time()

def load_raw_data():
    script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the script
    covid_path = os.path.join(script_dir, "data", "covid")
    non_covid_path = os.path.join(script_dir, "data", "noncovid")

    covid_images = list(Path(covid_path).glob("*.png"))
    non_covid_images = list(Path(non_covid_path).glob("*.png"))

    # covid_images is about 7500 while non_covid_images is about 7000
    return covid_images[:2000], non_covid_images[:2000]


covid_images, non_covid_images = load_raw_data()
IMG_SIZE = 128

# Two empty numpy arrays to store coverted images
positive_npy = np.empty(
    (len(covid_images), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
negative_npy = np.empty(
    (len(non_covid_images), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

# start a bar of show percentage of loading data
covid_bar = progressbar.ProgressBar(maxval=len(covid_images), widgets=[
                                    progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
non_covid_bar = progressbar.ProgressBar(maxval=len(non_covid_images), widgets=[
                                        progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

covid_bar.start()
# Converting COVID dataset to .npy format
for i, _file in enumerate(covid_images):
    try:
        image_npy = img.imread(_file)
        positive = resize(image_npy, (IMG_SIZE, IMG_SIZE, 1),
                          anti_aliasing=True)
        positive_npy[i] = positive
    except:
        pass
    covid_bar.update(i+1)

covid_bar.finish()
print("COVID images converting done")

non_covid_bar.start()
# Converting non-COVID dataset to .npy format
for i, _file in enumerate(non_covid_images):
    try:
        image_npy = img.imread(_file)
        negative = resize(image_npy, (IMG_SIZE, IMG_SIZE, 1),
                          anti_aliasing=True)
        negative_npy[i] = negative
    except:
        pass

    non_covid_bar.update(i+1)
non_covid_bar.finish()
print("non covid images converting done")


def load_data():
    """To load dataset and concat them to x and y. """
    positive = positive_npy
    positive_labels = ["1" for i in positive]
    negative = negative_npy
    negative_labels = ["0" for i in negative]

    # Joining both datasets and labels
    X = np.concatenate([positive, negative])
    y = np.array((positive_labels + negative_labels), dtype=np.float32)
    return X, y


# CNN model
def cnn():
    """
    This function creates and compiles a cnn model using the tensorflow keras api 
    First three convolutional layers, each followed by batch normalization and max pooling. 
        - each layer uses 5x5 filters and a ReLU activation function, with L2 regularization. 
        - first, second, and third layers contain 32, 16, and 32 filters respectively.
    After convolutional layers we have a Flatten layer to convert 3D output to 1D
    After is a fully connected Dense layer with 200 nodes
        - uses a ReLU activation function and L2 regularization. 
        - has a dropout rate of 0.5 to prevent overfitting.
    Final layer also a Dense layer with 2 nodes to representing the two output classes.
        - uses a softmax activation function for multiclass classification
    Model is compiled with Adam Optimizer
    """
    model = Sequential()
    # convulutional layer
    model.add(
        Conv2D(
            32,
            kernel_size=5,
            activation="relu",
            input_shape=(128, 128, 1),
            kernel_regularizer=regularizers.l2(0.01)
        )
    )
    # Normalising after activation
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=5, activation="relu",
                     kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=5, activation="relu",
                     kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    # fully connected layer
    model.add(Dense(200, activation="relu",
                    kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))

    # output
    model.add(Dense(2, activation="softmax"))
    model.compile(
        loss=keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0),
        metrics=["accuracy"],
    )
    return model


def measure_memory():
    """
    Trains a local CNN model on COVID-19 images while tracking memory usage and execution time.
    """
    start_time = time.time()

    covid_images, non_covid_images = load_raw_data()
    IMG_SIZE = 128
    # Two empty numpy arrays to store converted images
    positive_npy = np.empty(
        (len(covid_images), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    negative_npy = np.empty(
        (len(non_covid_images), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    X, y = load_data()
    print("Dataset shape:")
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    model = cnn()

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.05,
        patience=2,
        verbose=1,
        restore_best_weights=True)

    model.summary()
    
    history_cnn = model.fit(
        X_train,
        y_train,
        epochs=45,
        workers=4,
        validation_data=(X_val, y_val),
    )

    y_pred = model.predict(X_test)
    predicted = np.argmax(y_pred, axis=-1)
    accuracy = np.equal(y_test, predicted).mean()
    print("Accuracy:", accuracy)

    print(classification_report(y_test, predicted))

    # Calculate the execution time
    execution_time = time.time() - start_time

    # Print the execution time
    print("Execution time:", execution_time)

mem_usage = memory_usage(measure_memory)
print("Memory usage (in MB):", max(mem_usage))