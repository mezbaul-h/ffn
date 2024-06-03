import os
import tempfile

from ffn.utils import initialize_xavier_weights, make_zeroes_matrix, read_dataset_csv, train_test_split


def test_initialize_xavier_weights():
    weights = initialize_xavier_weights(3, 2, random_state=42)
    assert len(weights) == 3
    assert len(weights[0]) == 2
    assert all(isinstance(weight, float) for row in weights for weight in row)


def test_make_zeroes_matrix():
    zero_matrix = make_zeroes_matrix(2, 2)
    assert zero_matrix == [[0.0, 0.0], [0.0, 0.0]]


def test_read_dataset_csv():
    data = """1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0"""
    with tempfile.NamedTemporaryFile(delete=False, mode="w", newline="") as temp_file:
        temp_file.write(data)
        temp_file_path = temp_file.name

    features, outputs = read_dataset_csv(temp_file_path)
    os.remove(temp_file_path)

    assert features == [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]
    assert outputs == [[3.0], [6.0], [9.0]]


def test_read_dataset_csv_with_transformer():
    data = """1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0"""
    with tempfile.NamedTemporaryFile(delete=False, mode="w", newline="") as temp_file:
        temp_file.write(data)
        temp_file_path = temp_file.name

    def transformer(rows):
        return [row for row in rows if row[0] > 1.0]

    features, outputs = read_dataset_csv(temp_file_path, transformer=transformer)
    os.remove(temp_file_path)

    assert features == [[4.0, 5.0], [7.0, 8.0]]
    assert outputs == [[6.0], [9.0]]


def test_train_test_split():
    features = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    outputs = [[1], [0], [1], [0], [1]]
    features_train, features_test, outputs_train, outputs_test = train_test_split(
        features, outputs, test_size=0.4, random_state=42
    )

    assert len(features_train) == 3
    assert len(features_test) == 2
    assert len(outputs_train) == 3
    assert len(outputs_test) == 2
    assert features_train != features_test
    assert outputs_train != outputs_test
