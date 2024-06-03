import json

import pytest

from ffn.scalers import MinMaxScaler


@pytest.fixture
def sample_data():
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


@pytest.fixture
def scaler(sample_data):
    return MinMaxScaler(sample_data)


def test_initialization(scaler):
    assert scaler.column_mins == [1, 2, 3]
    assert scaler.column_maxes == [7, 8, 9]


def test_get_params(scaler):
    params = scaler.get_params()
    assert params["column_mins"] == [1, 2, 3]
    assert params["column_maxes"] == [7, 8, 9]


def test_load_params_from_dict(scaler):
    new_params = {"column_mins": [0, 0, 0], "column_maxes": [10, 10, 10]}
    scaler.load_params(new_params)
    assert scaler.column_mins == [0, 0, 0]
    assert scaler.column_maxes == [10, 10, 10]


def test_load_params_from_json_file(tmp_path, scaler):
    new_params = {"column_mins": [0, 0, 0], "column_maxes": [10, 10, 10]}
    json_file = tmp_path / "params.json"
    json_file.write_text(json.dumps(new_params))

    scaler.load_params(str(json_file))
    assert scaler.column_mins == [0, 0, 0]
    assert scaler.column_maxes == [10, 10, 10]


def test_transform(scaler):
    data = [[2, 3, 4]]
    transformed_data = scaler.transform(data)
    assert transformed_data == [[0.16666666666666666, 0.16666666666666666, 0.16666666666666666]]


def test_transform_with_zero_division():
    scaler = MinMaxScaler([[1], [1]])
    data = [[1], [1]]
    transformed_data = scaler.transform(data)
    assert transformed_data == [[0.0], [0.0]]


def test_inverse_transform(scaler):
    scaled_data = [[0.16666666666666666, 0.16666666666666666, 0.16666666666666666]]
    inverse_transformed_data = scaler.inverse_transform(scaled_data)
    assert inverse_transformed_data == [[2.0, 3.0, 4.0]]


def test_inverse_transform_with_zero_division():
    scaler = MinMaxScaler([[1], [1]])
    scaled_data = [[0.0], [0.0]]
    inverse_transformed_data = scaler.inverse_transform(scaled_data)
    assert inverse_transformed_data == [[1.0], [1.0]]
