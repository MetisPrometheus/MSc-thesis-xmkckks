# Standard Libraries
import numpy as np

# Third Party Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dropout, Dense, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from sklearn.metrics import log_loss
# import torch



class CNN():
    def __init__(self, weight_decimals=8):
        self.model = None
        self.WEIGHT_DECIMALS = self.set_weight_decimals(weight_decimals)

    def set_weight_decimals(self, weight_decimals):
        if 2 <= weight_decimals <= 8:
            return weight_decimals
        return 8

    def set_initial_params(self):
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
        self.model = model

    def fit(self, X_train, y_train, X_val, y_val, epochs=20, workers=4):
        """
        Takes in epochs and workers and starts training the model 
        ----------
        epochs : int
            The number of epochs in training
        workers : int
            The number of workers in the training
        """
        tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2)

        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.05,
            patience=2,
            verbose=1,
            restore_best_weights=True)

        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            workers=workers,
            validation_data=(X_val, y_val),
        )

    def get_weights(self):
        """
        returns the weights of the model
        """
        # return self.model.get_weights()
        weights = self.model.get_weights()
        scaled_weights = [tf.math.round(w * 10**self.WEIGHT_DECIMALS) for w in weights]
        return scaled_weights

    def set_weights(self, parameters):
        """
        Sets the weight of the model
        """
        # self.model.set_weights(parameters)
        scaled_weights = [w / 10**self.WEIGHT_DECIMALS for w in parameters]
        self.model.set_weights(scaled_weights)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the accuracy and gives a classification report of the result from the CNN model
        """
        y_pred = self.model.predict(X_test)
        predicted = np.argmax(y_pred, axis=-1)
        accuracy = np.equal(y_test, predicted).mean()
        loss = log_loss(y_test, y_pred)

        return loss, accuracy
    

    def flatten_list(self, nested_list):
        """
        Takes nested list of tensors from cnn models and flattens it into a single list of elements.
        """
        flattened = []
        for item in nested_list:
            #if torch.is_tensor(item):
            #    flattened.extend(self.flatten_list(item.tolist() if isinstance(item, np.ndarray) else item))
            if isinstance(item, (list, np.ndarray)):
                flattened.extend(self.flatten_list(item.tolist() if isinstance(item, np.ndarray) else item))
            else:
                flattened.append(item)
        return flattened

    def unflatten_list(self, flat_list, shapes):
        """
        Reshapes flattened list back to a nested list of tensors for a cnn model.
        """
        unflattened_list = []
        index = 0
        for shape in shapes:
            size = np.prod(shape)
            unflattened_list.append(np.array(flat_list[index:index + size]).reshape(shape))
            index += size
        return unflattened_list