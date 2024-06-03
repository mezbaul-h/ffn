"""
Scaler Module

This module provides classes for scaling numerical data to specified ranges.
"""

import json


class MinMaxScaler:
    """
    A class for scaling numerical data to a specified range (default: [0, 1]).

    Attributes
    ----------
    data : list
        The input data used for computing scaling parameters.
    column_mins : list
        A list containing the minimum values for each column in the input data.
    column_maxes : list
        A list containing the maximum values for each column in the input data.
    """

    def __init__(self, data=None):
        """
        Initialize the MinMaxScaler.

        Parameters
        ----------
        data : list, optional, default: None
            The input data to compute scaling parameters. If not provided, an empty list is used.
        """
        self.data = data or []
        columns = [list(column) for column in zip(*self.data)]
        self.column_mins = [min(column) for column in columns]
        self.column_maxes = [max(column) for column in columns]

    def get_params(self):
        """
        Get the current scaling parameters.

        Returns
        -------
        dict
            A dictionary containing the current scaling parameters.
        """
        params = ["column_mins", "column_maxes"]

        return {param: getattr(self, param) for param in params}

    def load_params(self, source):
        """
        Load scaling parameters from a dictionary, a JSON file, or a
        pathlib.Path object.

        Parameters
        ----------
        source : dict, str, or pathlib.Path
            If a dictionary, it should contain scaling parameters.
            If a string, it is treated as a file path to a JSON file containing scaling parameters.
            If a pathlib.Path object, it is treated as the path to a JSON file containing scaling parameters.
        """
        params = ["column_mins", "column_maxes"]

        if not isinstance(source, dict):
            with open(source, "r") as f:
                source = json.loads(f.read())

        for param in params:
            setattr(self, param, source[param])

    def transform(self, data):
        """
        Scale input data to the specified range.

        Parameters
        ----------
        data : list
            The data to be scaled.

        Returns
        -------
        list
            The scaled data.
        """
        transformed_data = []

        for row in data:
            transformed_row = []

            for i in range(len(row)):
                try:
                    # Scale each element to the range [0, 1].
                    transformed_row.append(
                        (row[i] - self.column_mins[i]) / (self.column_maxes[i] - self.column_mins[i])
                    )
                except ZeroDivisionError:
                    # Handle the case where the denominator is zero.
                    transformed_row.append(0.0)

            transformed_data.append(transformed_row)

        return transformed_data

    def inverse_transform(self, data):
        """
        Inverse scale transformed data back to the original range.

        Parameters
        ----------
        data : list
            The scaled data to be inverse transformed.

        Returns
        -------
        list
            The data in the original scale.
        """
        inverse_transformed_data = []

        for row in data:
            inverse_transformed_data.append(
                [
                    # Inverse scale each element back to the original range
                    (row[i] * (self.column_maxes[i] - self.column_mins[i])) + self.column_mins[i]
                    for i in range(len(row))
                ]
            )

        return inverse_transformed_data
