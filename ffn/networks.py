"""
Neural Network Container Classes.

This module contains classes for neural network containers. A neural network container manages the architecture,
training, and evaluation of a neural network.
"""

import json
import random
import time

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from . import activations, layers
from .scalers import MinMaxScaler


class Sequential:
    """
    Container class for building a sequential neural network.

    This class allows the construction of a neural network by stacking linear layers sequentially. It provides methods
    for training, evaluation, saving, and loading the network.

    Attributes
    ----------
    learning_rate : float
        The learning rate for the network.
    momentum : float
        The momentum for updating weights during training.
    layers : list
        The list of layers in the network.
    num_layers : int
        Number of layers in the network.
    feature_scaler : ffn.scalers.MinMaxScaler
        Scaler for input features.
    output_scaler : ffn.scalers.MinMaxScaler
        Scaler for output targets.
    num_epochs : int
        The number of training epochs.
    current_epoch : int
        The current training epoch.
    epoch_losses : dict
        Dictionary to store training and validation losses for each epoch.
    layer_params_at_lowest_training_epoch : list
        The list of layer parameters at the epoch with the lowest training loss.
    layer_params_at_lowest_validation_epoch : list
        The list of layer parameters at the epoch with the lowest validation loss.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Sequential network.

        Parameters
        ----------
        *args : layers.Linear
            A list of Linear layers to be added to the sequential network.
        learning_rate : float
            The learning rate for the network.
        momentum : float
            The momentum for updating weights during training.
        feature_scaler : ffn.scalers.MinMaxScaler
            Scaler for input features.
        output_scaler : ffn.scalers.MinMaxScaler
            Scaler for output targets.
        num_epochs : int
            The number of training epochs.
        """
        self.learning_rate = kwargs.get("learning_rate")
        self.momentum = kwargs.get("momentum")

        self.layers = args
        self.num_layers = len(args)

        self._fix_layers()

        self.feature_scaler = kwargs.get("feature_scaler")
        self.output_scaler = kwargs.get("output_scaler")

        self.num_epochs = kwargs["num_epochs"]
        self.current_epoch = 0

        self.epoch_losses = {
            "training": [],
            "validation": [],
        }
        self.layer_params_at_lowest_training_epoch = [layer.get_params() for layer in self.layers]
        self.layer_params_at_lowest_validation_epoch = [layer.get_params() for layer in self.layers]

    def _fix_layers(self):
        """
        Fix connections and parameters of layers in the network.

        Adjusts the connections, learning rate, and momentum for each layer.
        """
        for i in range(self.num_layers):
            current_layer = self.layers[i]
            current_layer.next_layer = self.layers[i + 1] if i < (self.num_layers - 1) else None
            current_layer.previous_layer = self.layers[i - 1] if i > 0 else None

            current_layer.learning_rate = self.learning_rate
            current_layer.momentum = self.momentum

            if current_layer.activation:
                current_layer.activation.learning_rate = self.learning_rate

    def backward(self, losses):
        """
        Backward pass through the network.

        Parameters
        ----------
        losses : list
           Losses for each output dimension.
        """
        carry_over_gradient = None

        # Iterate through the layers in reverse order.
        for l_index in range(self.num_layers - 1, -1, -1):
            current_layer = self.layers[l_index]
            layer_gradients = []

            for i in range(current_layer.input_feature_count):
                # Calculate gradients for neurons in the previous layer if one exists.
                if current_layer.previous_layer:
                    layer_gradients.append(
                        sum(
                            [
                                losses[k] * current_layer.weights[i][k]
                                for k in range(current_layer.output_feature_count)
                            ]
                        )
                        * current_layer.previous_layer.activation.derivative(current_layer.inputs[i])
                    )

                # Update delta weights and biases for the current layer.
                for j in range(current_layer.output_feature_count):
                    if not current_layer.next_layer:  # Output layer
                        delta_bias = losses[j]
                        delta_weight = current_layer.inputs[i] * losses[j]
                    else:
                        delta_bias = carry_over_gradient[j]
                        delta_weight = current_layer.inputs[i] * carry_over_gradient[j]

                    current_layer.delta_weights[i][j] = delta_weight
                    current_layer.delta_biases[j] = delta_bias

            # Update carry-over gradient for the next iteration.
            carry_over_gradient = layer_gradients

        # Update weights and biases using deltas for each layer.
        for layer in self.layers:
            layer.update_biases_and_weights()

    def evaluate(self, x_test, y_test, use_best_layer_params=False):
        """
        Evaluate the performance of the network on a test set.

        Parameters
        ----------
        x_test : list
            The list of input features for the test set.
        y_test : list
            The list of target outputs for the test set.
        use_best_layer_params : bool, optional
            If True, use the best layer parameters for evaluation.

        Returns
        -------
        list
            The list of average evaluation losses for each output dimension.
        """
        old_layer_params = None

        # If using best layer params, record current layer states and load the best states
        if use_best_layer_params and self.layer_params_at_lowest_validation_epoch:
            old_layer_params = []

            for i, layer in enumerate(self.layers):
                old_layer_params.append(layer.get_params())
                layer.load_params(self.layer_params_at_lowest_validation_epoch[i])

        total_evaluation_losses = [0.0] * len(y_test[0])
        avg_evaluation_losses = total_evaluation_losses.copy()
        num_samples = len(x_test)

        for features, target_outputs in zip(x_test, y_test):
            # Validation features are not scaled, so that needs to go through the same transformation.
            scaled_features = self.feature_scaler.transform([features])[0]

            calculated_outputs = self.predict(scaled_features)

            # Target outputs are not scaled as well.
            scaled_target_outputs = self.output_scaler.transform([target_outputs])[0]

            losses = [prediction - target for prediction, target in zip(calculated_outputs, scaled_target_outputs)]

            for i in range(len(losses)):
                total_evaluation_losses[i] += losses[i] ** 2

        for i in range(len(total_evaluation_losses)):
            avg_evaluation_losses[i] = (total_evaluation_losses[i] / num_samples) ** 0.5

        # If using best layer params, reload old layer states.
        if old_layer_params:
            for i, layer in enumerate(self.layers):
                layer.load_params(old_layer_params[i])

        return avg_evaluation_losses

    def forward(self, features):
        """
        Forward pass through the network.

        Parameters
        ----------
        features : list
            Input features.

        Returns
        -------
        list
            Output predictions.
        """
        for index, layer in enumerate(self.layers):
            features = layer.forward(features)

        return features

    @staticmethod
    def get_best_epoch(epoch_losses, default):
        """
        Get the epoch index with the lowest loss.

        Parameters
        ----------
        epoch_losses : list
            A list of losses for each epoch.
        default : int
            Default epoch index.

        Returns
        -------
        int
            Index of the epoch with the lowest loss.
        """
        try:
            return epoch_losses.index(min(epoch_losses))
        except ValueError:
            return default

    def get_params(self):
        """
        Get the parameters of the network.

        Returns
        -------
        dict
            Dictionary containing the parameters of the network.
        """
        return {
            "current_epoch": self.current_epoch,
            "epoch_losses": self.epoch_losses,
            "feature_scaler_params": self.feature_scaler.get_params(),
            "layer_params": [layer.get_params() for layer in self.layers],
            "layer_params_at_lowest_training_epoch": self.layer_params_at_lowest_training_epoch,
            "layer_params_at_lowest_validation_epoch": self.layer_params_at_lowest_validation_epoch,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "num_epochs": self.num_epochs,
            "output_scaler_params": self.output_scaler.get_params(),
        }

    @classmethod
    def load(cls, checkpoint_filename):
        """
        Load a Sequential network from a saved checkpoint file.

        Parameters
        ----------
        checkpoint_filename : str
            Name of the checkpoint file.

        Returns
        -------
        Sequential
            Loaded Sequential network.
        """
        with open(checkpoint_filename, "r") as f:
            checkpoint_data = json.loads(f.read())

        network_layers = []

        # Iterate over the layer parameters in the checkpoint data.
        for layer_param in checkpoint_data["layer_params"]:
            activation_function = None

            # Check if the layer uses sigmoid activation.
            if layer_param["sigmoid_activation"]:
                activation_function = activations.Sigmoid()

            # Create a Linear layer with the specified parameters.
            layer = layers.Linear(
                layer_param["layer_dimensions"][0], layer_param["layer_dimensions"][1], activation_function
            )

            # Load weights, biases, and other parameters from the checkpoint.
            layer.weights = layer_param["weights"]
            layer.delta_weights = layer_param["delta_weights"]
            layer.biases = layer_param["biases"]
            layer.delta_biases = layer_param["delta_biases"]

            # Add the layer to the list of network layers.
            network_layers.append(layer)

        feature_scaler = MinMaxScaler()
        feature_scaler.load_params(checkpoint_data["feature_scaler_params"])

        output_scaler = MinMaxScaler()
        output_scaler.load_params(checkpoint_data["output_scaler_params"])

        # Create a Sequential instance with loaded parameters.
        instance = cls(
            *network_layers,
            feature_scaler=feature_scaler,
            learning_rate=checkpoint_data["learning_rate"],
            momentum=checkpoint_data["momentum"],
            num_epochs=checkpoint_data["num_epochs"],
            output_scaler=output_scaler,
        )

        # Load additional information and states from the checkpoint.
        instance.current_epoch = checkpoint_data["current_epoch"]
        instance.epoch_losses = checkpoint_data["epoch_losses"]
        instance.layer_params_at_lowest_training_epoch = checkpoint_data["layer_params_at_lowest_training_epoch"]
        instance.layer_params_at_lowest_validation_epoch = checkpoint_data["layer_params_at_lowest_validation_epoch"]

        return instance

    def predict(self, features):
        """
        Make predictions using the network.

        Parameters
        ----------
        features : list
            Input features.

        Returns
        -------
        list
            Output predictions.
        """
        predictions = self.forward(features)

        return predictions

    def save(self, checkpoint_filename="checkpoint.json"):
        """
        Save the network parameters to a file.

        Parameters
        ----------
        checkpoint_filename : str, optional
            Name of the file to save the parameters.
        """
        with open(checkpoint_filename, "w+") as f:
            f.write(json.dumps(self.get_params(), indent=4))

    def save_loss_plot(self, target_filename="loss_plot.png"):  # pragma: no cover
        """
        Save a plot of training and validation losses.

        Parameters
        ----------
        target_filename : str, optional
            Name of the file to save the plot.
        """
        # Don't try to plot if matplotlib is not available.
        if not plt:
            return

        # Extract losses for training and validation.
        training_losses = self.epoch_losses["training"]
        best_training_epoch_index = self.get_best_epoch(training_losses, self.current_epoch)
        validation_losses = self.epoch_losses["validation"]
        best_validation_epoch_index = self.get_best_epoch(validation_losses, self.current_epoch)

        # Set the figure size to 1000 x 600 pixels.
        plt.figure(figsize=(10, 6))

        # Create training line plot.
        plt.plot(range(1, len(training_losses) + 1), training_losses, color="red", label="Training Loss")
        plt.axvline(
            x=best_training_epoch_index + 1, color="red", label=f"Best Training Epoch: {best_training_epoch_index + 1}"
        )

        # Create validation line plot.
        plt.plot(range(1, len(validation_losses) + 1), validation_losses, color="green", label="Validation Loss")
        plt.axvline(
            x=best_validation_epoch_index + 1,
            color="green",
            label=f"Best Validation Epoch: {best_validation_epoch_index + 1}",
        )

        # Set labels and title.
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            "Training and Validation Loss Over Epochs\n"
            f"Learning Rate: {self.learning_rate}, Momentum: {self.momentum}"
        )

        # Show the legend.
        plt.legend()

        # Save the plot to an image file.
        plt.savefig(target_filename)

    def train(self, x_train, y_train, x_validation, y_validation, log_prefix=None):
        """
        Train the network on the provided datasets.

        Parameters
        ----------
        x_train : list
            A list of input features for the training set.
        y_train : list
            A list of target outputs for the training set.
        x_validation : list
            A list of input features for the validation set.
        y_validation : list
            A list of target outputs for the validation set.
        log_prefix : str, optional
            Prefix for log messages during training.
        """
        num_samples = len(x_train)
        training_index_order = list(range(len(x_train)))

        while self.current_epoch < self.num_epochs:
            epoch_start_time = time.time()

            total_training_losses = [0.0] * len(y_train[0])
            avg_training_losses = total_training_losses.copy()

            # Iterate through training samples.
            for index in training_index_order:
                features = x_train[index]
                targets = y_train[index]

                # Forward pass to get predictions.
                predictions = self.forward(features)

                # Calculate losses.
                losses = [prediction - target for prediction, target in zip(predictions, targets)]

                # Accumulate losses for each output dimension.
                for i in range(len(losses)):
                    total_training_losses[i] += losses[i] ** 2

                # Backward pass to update weights and biases.
                self.backward(losses)

            # Calculate average training losses.
            for i in range(len(avg_training_losses)):
                avg_training_losses[i] = (total_training_losses[i] / num_samples) ** 0.5

            # Evaluate the network on the validation set.
            avg_validation_losses = self.evaluate(x_validation, y_validation)

            mean_training_loss = sum(avg_training_losses) / len(avg_training_losses)
            mean_validation_loss = sum(avg_validation_losses) / len(avg_validation_losses)

            # Record losses for plotting.
            self.epoch_losses["training"].append(mean_training_loss)
            self.epoch_losses["validation"].append(mean_validation_loss)

            # Get the best epochs for training.
            best_training_epoch_index = self.get_best_epoch(self.epoch_losses["training"], self.current_epoch)

            # Save layer parameters (weights, biases) if current epoch is the best training epoch.
            if best_training_epoch_index == self.current_epoch:
                self.layer_params_at_lowest_training_epoch = [layer.get_params() for layer in self.layers]

            # Get the best epochs for validation.
            best_validation_epoch_index = self.get_best_epoch(self.epoch_losses["validation"], self.current_epoch)

            # Save layer parameters (weights, biases) if current epoch is the best validation epoch.
            if best_validation_epoch_index == self.current_epoch:
                self.layer_params_at_lowest_validation_epoch = [layer.get_params() for layer in self.layers]

            seconds_elapsed = time.time() - epoch_start_time

            # Print training progress.
            print(
                f"{log_prefix or ''}"
                f"[{self.current_epoch + 1}/{self.num_epochs}] "
                f"Losses (T/V): {mean_training_loss:.10f}/{mean_validation_loss:.10f} | "
                f"Best Epochs (T/V): {best_training_epoch_index + 1}/{best_validation_epoch_index + 1} "
                f"({seconds_elapsed:.2f}s)"
            )

            # Move to the next epoch.
            self.current_epoch += 1

            # Randomize the training indices for the next epoch.
            random.shuffle(training_index_order)
