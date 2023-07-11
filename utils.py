# Standard Libraries
import importlib
import os
import sys
from typing import List, Tuple, Union

# Thirs Party Imports
import numpy as np
import openml
import tensorflow as tf

# Local Imports
# Get absolute paths to let a user run the script from anywhere
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.basename(current_directory)
working_directory = os.getcwd()
# Add parent directory to Python's module search path
sys.path.append(os.path.join(current_directory, '..'))
# Compare paths
if current_directory == working_directory:
    from cnn import CNN
else:
    # Add current directory to Python's module search path
    CNN = importlib.import_module(f"{parent_directory}.cnn").CNN

# Define datatypes
XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]



def set_initial_params(model: CNN):
    """Sets initial parameters for CNN """
    model.set_initial_params()


def get_model_parameters(model: CNN) -> LogRegParams:
    """Returns the paramters of a CNN model. """
    params = model.get_weights()
    return params


def set_model_params(model: CNN, params: LogRegParams) -> CNN:
    """Sets the parameters of a CNN model. """
    model.set_weights(params)
    return model


def get_flat_weights(model: CNN) -> Tuple[List[int], List[int]]:
    """ Get model weights, convert and flatten nested tensor to a list, original shape is saved. """
    params = model.get_weights()
    original_shapes = [tensor.shape for tensor in params]
    flattened_weights = []
    for weight_tensor in params:
        # Convert the tensor to a numpy array
        weight_array = weight_tensor.numpy()
        # Flatten the numpy array and add it to the flattened_weights list
        flattened_weights.extend(weight_array.flatten())
    return flattened_weights, original_shapes


def unflatten_weights(flattened_weights, original_shapes):
    """Convert flat list into the model's original nested tensors. """
    unflattened_weights = []
    current_index = 0
    for shape in original_shapes:
        # Calculate the number of elements in the current shape
        num_elements = np.prod(shape)
        # Slice the flattened_weights list to get the elements for the current shape
        current_elements = flattened_weights[current_index : current_index + num_elements]
        # Reshape the elements to the original shape and append them to the unflattened_weights list
        reshaped_elements = np.reshape(current_elements, shape)
        unflattened_weights.append(tf.convert_to_tensor(reshaped_elements, dtype=tf.float64))
        # Update the index for the next iteration
        current_index += num_elements
    return unflattened_weights


def pad_to_power_of_2(flat_params, target_length=2**20, weight_decimals=8):
    """
    - Client Side
    - Before encryption of local weights:
    - Pad flat weights to nearest 2^n, original length is saved. 
    """
    pad_length = target_length - len(flat_params)
    if pad_length < 0:
        raise ValueError("The given target_length is smaller than the current parameter list length.")
    # Let the padding be random numbers within the min and max values of the weights
    random_padding = np.random.randint(-10**weight_decimals, 10**weight_decimals + 1, pad_length).tolist()
    padded_params = flat_params + random_padding
    return padded_params, len(flat_params)


def remove_padding(padded_params, original_length):
    """
    - Client Side
    - After receiving decrypted model updates:
    - Pad flat weights to nearest 2^n, original length is saved. 
    """
    return padded_params[:original_length]


def load_mnist() -> Dataset:
    """
    Loads the MNIST dataset using OpenML.
    OpenML dataset link: https://www.openml.org/d/554
    """
    mnist_openml = openml.datasets.get_dataset(554)
    Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    X = Xy[:, :-1]  # the last column contains labels
    y = Xy[:, -1]
    # First 60000 samples consist of the train set
    x_train, y_train = X[:60000], y[:60000]
    x_test, y_test = X[60000:], y[60000:]
    return (x_train, y_train), (x_test, y_test)


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y. """
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions. """
    return list(
        zip(np.array_split(X, num_partitions),
            np.array_split(y, num_partitions))
    )


def is_prime(n):
    """Check if n is a prime number. """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def next_prime(n):
    """Find the smallest prime number larger than n. """
    if n <= 1:
        return 2
    prime = n
    found = False
    while not found:
        prime += 1
        if is_prime(prime):
            found = True
    return prime