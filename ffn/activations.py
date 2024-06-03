"""
Activations Module

This module provides activation functions for neural networks.
"""

import math


class Sigmoid:
    """
    Sigmoid activation function.

    Attributes
    ----------
    inputs : array-like
        Input values.
    learning_rate : None
        Not used in the Sigmoid activation function.
    outputs : list
        Output values after activation.
    """

    def __init__(self):
        """
        Initialize the Sigmoid activation function.
        """
        self.inputs = None
        self.learning_rate = None
        self.outputs = None

    @staticmethod
    def activate(value):
        """
        Apply the sigmoid activation function.

        Parameters
        ----------
        value : float
            The input value.

        Returns
        -------
        float
            The activated output value.
        """
        try:
            return 1 / (1 + math.exp(-value))
        except OverflowError:
            # Handle overflow errors gracefully.
            return 1 - (1 / 1 + math.exp(value))

    @staticmethod
    def derivative(value):
        """
        Calculate the derivative of the sigmoid activation function.

        Parameters
        ----------
        value : float
            The input value.

        Returns
        -------
        float
            The derivative of the sigmoid activation function.
        """
        return value * (1 - value)

    def forward(self, x):
        """
        Perform forward pass through the Sigmoid activation layer.

        Parameters
        ----------
        x : array-like
            Input values.

        Returns
        -------
        list
            Output values after activation.
        """
        self.inputs = x
        self.outputs = []

        for item in x:
            self.outputs.append(self.activate(item))

        return self.outputs
