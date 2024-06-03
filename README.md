# Feed-Forward Neural Network with Backpropagation

This repository features a pure Python implementation of a Feed-Forward neural network with Backpropagation. It serves as the foundation for training shallow deep learning models for simple problems.

### Features:

- Flexible network components including layers, activation functions, and feature scalers.
- Easily extensible to accommodate various types of layers, network architectures, activations, and scalers.
- Checkpoint support for seamless and hassle-free network training.

## INSTALLATION

### Pre-requisites:
- Python `3.10+` (with _pip_ and _setuptools_)
- Ubuntu `22.04+`

_NOTE_: These are not strict pre-requisites. The project was built and tested on Python `3.10.14` and Ubuntu `24.04 LTS`, so it should run on any Debian-based OS with reasonably recent versions of Python 3.

### Steps
1. Clone the project and go into the directory:
    ```shell
    git clone https://github.com/mezbaul-h/ffn.git
    cd ffn
    ```
2. Install the package:
    ```shell
    pip install .
    ```

You are now ready to use the package!


## USAGE

This library can be integrated into your code by simply importing it:

```python
import ffn

# do some stuff
```

### Creating a Sequential Neural Network

Below is an example demonstrating how to use this library to create a sequential network for solving the XOR problem. It showcases combinations of layers and activations.

```python
from ffn.networks import Sequential
from ffn.layers import Linear
from ffn.activations import Sigmoid
from ffn.scalers import MinMaxScaler

# Define the XOR problem datasets
x_train = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
y_train = [
    [0],
    [1],
    [1],
    [0]
]

# Define the architecture of the neural network
input_features = 2
output_features = 1

# Create the layers with specified activations
hidden_layer = Linear(input_features, 3, activation=Sigmoid(), random_state=43)
output_layer = Linear(3, output_features, random_state=43)

# Initialize feature and output scalers
feature_scaler = MinMaxScaler(x_train)
output_scaler = MinMaxScaler(y_train)

# Initialize the network container
network = Sequential(
    hidden_layer,
    output_layer,
    learning_rate=0.01,
    momentum=0.99,
    feature_scaler=feature_scaler,
    output_scaler=output_scaler,
    num_epochs=500,
)

# Validation dataset (using the same data for simplicity)
x_validation = x_train
y_validation = y_train

# Train the network
network.train(x_train, y_train, x_validation, y_validation)

# Save the trained network
network.save("trained_network.json")

# Test dataset (using the same data for simplicity)
x_test = x_train
y_test = y_train

# Test the network
for feature in x_test:
    # Transform the input feature with the scaler first
    prediction_scaled = network.predict(network.feature_scaler.transform([feature])[0])

    # Network returns scaled prediction, so run inverse transformation to unscale
    prediction = network.output_scaler.inverse_transform([prediction_scaled])[0]
    print(f"Input: {feature}, Prediction: {prediction}")
```

### Loading a Trained Neural Network

Below is an example demonstrating how to reload a model from a checkpoint and use it for making predictions:

```python
from ffn.networks import Sequential

# Load the trained network from a saved checkpoint
network = Sequential.load("trained_network.json")

# Define the XOR problem datasets
x_train = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
y_train = [
    [0],
    [1],
    [1],
    [0]
]

# Test dataset (using the same data for simplicity)
x_test = x_train
y_test = y_train

# Test the network
for feature in x_test:
    # Transform the input feature with the scaler first
    prediction_scaled = network.predict(network.feature_scaler.transform([feature])[0])

    # Network returns scaled prediction, so run inverse transformation to unscale
    prediction = network.output_scaler.inverse_transform([prediction_scaled])[0]
    print(f"Input: {feature}, Prediction: {prediction}")
```


## TESTING

To install the required package dependencies for testing, use the following command:

```shell
pip install ".[testing]"
```
After installing the dependencies, you can run the tests by executing:

```shell
make test
```

If you do not have `make` installed, you can run the tests using:

```shell
bash scripts/test.sh
```
