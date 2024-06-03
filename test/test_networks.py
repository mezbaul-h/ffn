from ffn.activations import Sigmoid
from ffn.layers import Linear
from ffn.networks import Sequential
from ffn.scalers import MinMaxScaler


def test_xor_problem():
    # Define the XOR problem
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [[0], [1], [1], [0]]

    # Initialize feature and output scalers
    feature_scaler = MinMaxScaler(x_train)
    x_train_scaled = feature_scaler.transform(x_train)

    output_scaler = MinMaxScaler(y_train)
    y_train_scaled = output_scaler.transform(y_train)

    # Create a Sequential network with 2 hidden layers
    network = Sequential(
        Linear(input_feature_count=2, output_feature_count=2, activation=Sigmoid()),
        Linear(input_feature_count=2, output_feature_count=1, activation=Sigmoid()),
        learning_rate=0.1,
        momentum=0.9,
        feature_scaler=feature_scaler,
        output_scaler=output_scaler,
        num_epochs=500,
    )

    # Train the network on the XOR problem
    network.train(x_train_scaled, y_train_scaled, x_train_scaled, y_train_scaled)

    # Test the trained network
    x_test = x_train
    y_test = y_train

    y_pred_scaled = [network.predict(feature_scaler.transform([x])[0]) for x in x_test]
    y_pred = output_scaler.inverse_transform(y_pred_scaled)

    # Check if the predictions are close to the expected values
    for pred, expected in zip(y_pred, y_test):
        assert round(pred[0]) == expected[0]
