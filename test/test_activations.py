import math

import pytest

from ffn.activations import Sigmoid


def test_sigmoid_activate():
    sigmoid = Sigmoid()

    # Test positive value
    assert sigmoid.activate(0.5) == pytest.approx(1 / (1 + math.exp(-0.5)), rel=1e-6)

    # Test negative value
    assert sigmoid.activate(-0.5) == pytest.approx(1 / (1 + math.exp(0.5)), rel=1e-6)

    # Test large positive value to check for overflow handling
    assert sigmoid.activate(100) == pytest.approx(1.0, rel=1e-6)

    # Test large negative value to check for overflow handling
    assert sigmoid.activate(-100) == pytest.approx(0.0, rel=1e-6)


def test_sigmoid_derivative():
    sigmoid = Sigmoid()

    # Test derivative at 0.5 (expected value: 0.25)
    assert sigmoid.derivative(0.5) == pytest.approx(0.25, rel=1e-6)

    # Test derivative at other values
    assert sigmoid.derivative(0.8) == pytest.approx(0.8 * (1 - 0.8), rel=1e-6)
    assert sigmoid.derivative(0.2) == pytest.approx(0.2 * (1 - 0.2), rel=1e-6)


def test_sigmoid_forward():
    sigmoid = Sigmoid()

    inputs = [0.5, -0.5, 1.0, -1.0, 0.0]
    expected_outputs = [sigmoid.activate(x) for x in inputs]

    assert sigmoid.forward(inputs) == pytest.approx(expected_outputs, rel=1e-6)

    # Check if inputs and outputs are set correctly
    assert sigmoid.inputs == inputs
    assert sigmoid.outputs == expected_outputs
