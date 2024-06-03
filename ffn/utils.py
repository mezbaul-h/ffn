"""
Utility Functions for Neural Network Operations

This module provides various utility functions for neural network operations, including weight initialization, matrix
creation, dataset reading, and data splitting.
"""

import csv
import math
import random


def _get_zero_weight():
    """
    Generate a zero weight value.

    Returns
    -------
    float
        Zero weight value.
    """
    return 0.0


def _make_matrix(row_size, column_size, value_func):
    """
    Create a matrix using a specified value generation function.

    Parameters
    ----------
    row_size : int
        Number of rows in the matrix.
    column_size : int
        Number of columns in the matrix.
    value_func : callable
        Function to generate values for each matrix element.

    Returns
    -------
    list of lists
        Matrix with specified dimensions and values.
    """
    return [[value_func() for _ in range(column_size)] for _ in range(row_size)]


def initialize_xavier_weights(input_size, output_size, random_state=None):
    """
    Initialize weights using the Xavier/Glorot Initialization.

    Parameters
    ----------
    input_size : int
        Number of input units.
    output_size : int
        Number of output units.
    random_state : int or None, optional
        Seed for reproducibility. If specified, the random number generator will be seeded for consistent results.
        Default is None.

    Returns
    -------
    weights : list
        Initialized weights with shape (input_size, output_size).

    Notes
    -----
    Xavier/Glorot Initialization scales the weights based on the number of input and output units to prevent
    vanishing/exploding gradients during training.

    The formula for standard deviation (std_dev) is:
        std_dev = sqrt(2 / (input_size + output_size))
    """
    if random_state is not None:
        random.seed(random_state)

    variance = 2.0 / (input_size + output_size)
    std_dev = math.sqrt(variance)

    weights = [[random.gauss(0, std_dev) for _ in range(output_size)] for _ in range(input_size)]

    return weights


def make_zeroes_matrix(row_size, column_size):
    """
    Create a matrix filled with zero weights.

    Parameters
    ----------
    row_size : int
        Number of rows in the matrix.
    column_size : int
        Number of columns in the matrix.

    Returns
    -------
    list of lists
        Matrix filled with zero weights.
    """
    return _make_matrix(row_size, column_size, _get_zero_weight)


def read_dataset_csv(csv_filename, transformer=None):
    """
    Read a CSV file containing a dataset and extract features and outputs based on the provided criterion.

    Parameters
    ----------
    csv_filename : str or pathlib.Path
        The path to the CSV file. It can be either a string or a Path object.
    transformer : callable, optional
        A function provided for row filtering. If supplied, this function will be invoked with the entire list of rows
        and is expected to return a new list of rows.

    Returns
    -------
    tuple
        A tuple containing two lists:
        - List of features, where each feature is represented as a list of floats.
        - List of outputs, where each output is represented as a list of floats.
    """
    # Read CSV file and convert rows to float values.
    data = []

    with open(csv_filename, "r") as csv_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            if row:
                data.append([float(item) for item in row])

    # Apply transformer function if provided.
    if transformer:
        data = transformer(data)

    # Separate features and outputs.
    features = []
    outputs = []

    for row in data:
        features.append([float(item) for item in row[:2]])
        outputs.append([float(item) for item in row[2:]])

    return features, outputs


def train_test_split(features, outputs, test_size=0.2, random_state=None):
    """
    Split arrays or lists into random train and test subsets.

    Parameters
    ----------
    features : array-like or list
        The input data. Features to be split.
    outputs : array-like or list
        The labels or outputs associated with the input data.
    test_size : float, optional, default: 0.2
        Proportion of the dataset to include in the test split.
    random_state : int or None, optional, default: None
        Seed for random number generation. If None, a random seed will be used.

    Returns
    -------
    tuple
        A tuple containing four arrays or lists:
        - `features_train`: Features for training.
        - `features_test`: Features for testing.
        - `outputs_train`: Labels or outputs for training.
        - `outputs_test`: Labels or outputs for testing.
    """
    if random_state is not None:
        random.seed(random_state)

    data = list(zip(features, outputs))
    random.shuffle(data)

    split_index = int(len(data) * (1 - test_size))

    train_data = data[:split_index]
    test_data = data[split_index:]

    features_train, outputs_train = zip(*train_data)
    features_test, outputs_test = zip(*test_data)

    return list(features_train), list(features_test), list(outputs_train), list(outputs_test)
