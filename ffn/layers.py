"""
Layers Module

This module provides classes for different layers in neural networks.
"""

from ffn.activations import Sigmoid
from ffn.utils import initialize_xavier_weights, make_zeroes_matrix


class Linear:
    """
    Represents a fully connected layer in a neural network.

    Attributes
    ----------
    input_feature_count : int
        Number of input features.
    output_feature_count : int
        Number of output features.
    inputs : list
        Input values during forward pass.
    learning_rate : None
        Learning rate for weight and bias updates.
    momentum : None
        Momentum term for updating weights and biases.
    outputs : list
        Output values after forward pass.
    biases : list
        Bias terms for each output feature.
    biases_momentum : list
        Momentum terms for biases.
    delta_biases : list
        Gradients of the biases.
    weights : list
        Weight matrix connecting input and output features.
    weights_momentum : list
        Momentum terms for weights.
    delta_weights : list
        Gradients of the weights.
    activation : Sigmoid or None
        Activation function for the layer. If None, no activation is applied.
    next_layer : Linear or None
        Reference to the next layer in the network. If None, this is the output layer.
    previous_layer : Linear or None
        Reference to the previous layer in the network. If None, this is the input layer.
    """

    def __init__(
        self,
        input_feature_count,
        output_feature_count,
        activation=None,
        random_state=None,
    ):
        """
        Initialize a fully connected layer.

        Parameters
        ----------
        input_feature_count : int
            Number of input features.
        output_feature_count : int
            Number of output features.
        activation : Sigmoid or None, optional
            Activation function for the layer. If None, no activation is applied. Default is None.
        random_state : int or None, optional
            Seed for reproducibility. If specified, the random number generator will be seeded for consistent weight
            initialization. Default is None.
        """
        self.input_feature_count = input_feature_count
        self.output_feature_count = output_feature_count
        self.inputs = None
        self.learning_rate = None
        self.momentum = None
        self.outputs = None

        self.biases = make_zeroes_matrix(1, output_feature_count)[0]
        self.biases_momentum = make_zeroes_matrix(1, output_feature_count)[0]
        self.delta_biases = make_zeroes_matrix(1, output_feature_count)[0]

        self.weights = initialize_xavier_weights(input_feature_count, output_feature_count, random_state=random_state)
        self.weights_momentum = make_zeroes_matrix(input_feature_count, output_feature_count)
        self.delta_weights = make_zeroes_matrix(input_feature_count, output_feature_count)

        self.activation = activation
        self.next_layer = None
        self.previous_layer = None

    def _calculate_momenta(self):
        """
        Update the momentum terms for biases and weights during training.
        """
        for i in range(self.output_feature_count):
            self.biases_momentum[i] = (self.momentum * self.biases_momentum[i]) - (
                self.learning_rate * self.delta_biases[i]
            )

        for i in range(self.input_feature_count):
            for j in range(self.output_feature_count):
                self.weights_momentum[i][j] = (self.momentum * self.weights_momentum[i][j]) - (
                    self.learning_rate * self.delta_weights[i][j]
                )

    def forward(self, features):
        """
        Perform forward pass through the layer.

        Parameters
        ----------
        features : list
            Input features for the layer.

        Returns
        -------
        list
            Output values after applying the layer's operation.
        """
        self.inputs = features
        self.outputs = [0] * self.output_feature_count

        for i in range(self.output_feature_count):
            self.outputs[i] = sum([features[j] * self.weights[j][i] for j in range(len(features))]) + self.biases[i]

        if self.activation:
            self.outputs = self.activation.forward(self.outputs)

        return self.outputs

    def get_params(self):
        """
        Get layer parameters as a dictionary.

        Returns
        -------
        dict
            Dictionary containing layer parameters.
        """
        return {
            "sigmoid_activation": self.activation is not None,
            "layer_dimensions": (self.input_feature_count, self.output_feature_count),
            "weights": self.weights,
            "delta_weights": self.delta_weights,
            "biases": self.biases,
            "delta_biases": self.delta_biases,
        }

    def load_params(self, param_dict):
        """
        Load layer parameters from a dictionary.

        Parameters
        ----------
        param_dict : dict
            Dictionary containing layer parameters.
        """
        self.biases = param_dict["biases"]
        self.delta_biases = param_dict["delta_biases"]
        self.delta_weights = param_dict["delta_weights"]
        self.input_feature_count, self.output_feature_count = param_dict["layer_dimensions"]
        self.weights = param_dict["weights"]

    def update_biases_and_weights(self):
        """
        Update biases and weights using the calculated momenta.
        """
        self._calculate_momenta()

        for i in range(self.output_feature_count):
            self.biases[i] += self.biases_momentum[i]

        for i in range(self.input_feature_count):
            for j in range(self.output_feature_count):
                self.weights[i][j] += self.weights_momentum[i][j]
