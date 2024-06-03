# test_layers.py

import pytest

from ffn.activations import Sigmoid
from ffn.layers import Linear
from ffn.utils import make_zeroes_matrix


def assert_nested_lists_almost_equal(actual, expected, rel_tol=1e-9):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a == pytest.approx(e, rel=rel_tol)


def fixed_weight_initialization(input_size, output_size):
    """
    Initialize weights with a fixed pattern for deterministic tests.
    """
    return [[0.1 * (i + j) for j in range(output_size)] for i in range(input_size)]


def test_linear_init():
    layer = Linear(input_feature_count=3, output_feature_count=2)
    layer.weights = fixed_weight_initialization(3, 2)

    assert layer.input_feature_count == 3
    assert layer.output_feature_count == 2
    assert layer.inputs is None
    assert layer.outputs is None
    assert layer.biases == [0, 0]
    assert layer.biases_momentum == [0, 0]
    assert layer.delta_biases == [0, 0]
    assert_nested_lists_almost_equal(layer.weights, fixed_weight_initialization(3, 2))
    assert_nested_lists_almost_equal(layer.weights_momentum, make_zeroes_matrix(3, 2))
    assert_nested_lists_almost_equal(layer.delta_weights, make_zeroes_matrix(3, 2))
    assert layer.activation is None
    assert layer.next_layer is None
    assert layer.previous_layer is None


def test_linear_forward():
    layer = Linear(input_feature_count=3, output_feature_count=2, activation=Sigmoid())
    layer.weights = fixed_weight_initialization(3, 2)
    features = [1.0, 2.0, 3.0]

    outputs = layer.forward(features)

    expected_outputs = [
        sum([features[j] * layer.weights[j][i] for j in range(len(features))]) + layer.biases[i]
        for i in range(layer.output_feature_count)
    ]
    expected_outputs = Sigmoid().forward(expected_outputs)

    assert_nested_lists_almost_equal(outputs, expected_outputs)
    assert layer.inputs == features
    assert_nested_lists_almost_equal(layer.outputs, expected_outputs)


def test_linear_get_params():
    layer = Linear(input_feature_count=3, output_feature_count=2, activation=Sigmoid())
    layer.weights = fixed_weight_initialization(3, 2)
    params = layer.get_params()

    expected_params = {
        "sigmoid_activation": True,
        "layer_dimensions": (3, 2),
        "weights": layer.weights,
        "delta_weights": layer.delta_weights,
        "biases": layer.biases,
        "delta_biases": layer.delta_biases,
    }

    assert params == expected_params


def test_linear_load_params():
    layer = Linear(input_feature_count=3, output_feature_count=2)
    param_dict = {
        "sigmoid_activation": False,
        "layer_dimensions": (3, 2),
        "weights": fixed_weight_initialization(3, 2),
        "delta_weights": make_zeroes_matrix(3, 2),
        "biases": [0, 0],
        "delta_biases": [0, 0],
    }

    layer.load_params(param_dict)

    assert layer.biases == param_dict["biases"]
    assert layer.delta_biases == param_dict["delta_biases"]
    assert_nested_lists_almost_equal(layer.delta_weights, param_dict["delta_weights"])
    assert layer.input_feature_count == param_dict["layer_dimensions"][0]
    assert layer.output_feature_count == param_dict["layer_dimensions"][1]
    assert_nested_lists_almost_equal(layer.weights, param_dict["weights"])


def test_linear_update_biases_and_weights():
    layer = Linear(input_feature_count=3, output_feature_count=2)
    layer.weights = fixed_weight_initialization(3, 2)
    layer.learning_rate = 0.1
    layer.momentum = 0.9

    layer.delta_biases = [0.5, -0.5]
    layer.delta_weights = [[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]]

    initial_biases = layer.biases[:]
    initial_weights = [row[:] for row in layer.weights]

    layer.update_biases_and_weights()

    expected_biases_momentum = [(0.9 * 0) - (0.1 * 0.5), (0.9 * 0) - (0.1 * -0.5)]
    expected_weights_momentum = [
        [(0.9 * 0) - (0.1 * 0.1), (0.9 * 0) - (0.1 * -0.1)],
        [(0.9 * 0) - (0.1 * 0.2), (0.9 * 0) - (0.1 * -0.2)],
        [(0.9 * 0) - (0.1 * 0.3), (0.9 * 0) - (0.1 * -0.3)],
    ]

    expected_biases = [initial_biases[i] + expected_biases_momentum[i] for i in range(layer.output_feature_count)]
    expected_weights = [
        [initial_weights[i][j] + expected_weights_momentum[i][j] for j in range(layer.output_feature_count)]
        for i in range(layer.input_feature_count)
    ]

    assert_nested_lists_almost_equal(layer.biases, expected_biases)
    assert_nested_lists_almost_equal(layer.weights, expected_weights)
    assert_nested_lists_almost_equal(layer.biases_momentum, expected_biases_momentum)
    assert_nested_lists_almost_equal(layer.weights_momentum, expected_weights_momentum)
