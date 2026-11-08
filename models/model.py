import torch
from torch import nn


class MLP(nn.Module):
    # one input layer, one hidden layer, one output layer
    def __init__(self, hidden_layers=[10]):
        super().__init__()

        layers = []
        # two inputs: (x1, x2)
        input_size = 2
        # fully connected layers
        for hidden_size in hidden_layers:
            # input layer -> hidden layer
            layers.append(nn.Linear(input_size, hidden_size))
            # activation function
            layers.append(nn.Tanh())
            # A way to connect hidden layer to the output layer
            input_size = hidden_size
        # hidden layer -> output layer
        layers.append(nn.Linear(input_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    model = MLP(hidden_layers=[10])
    print(model)

    # 3 samples for testing the forward pass
    X = torch.tensor([
        [0.0, 0.0],
        [0.5, 1.5],
        [-1.0, 2.0]
    ], dtype=torch.float32)

    predictions = model(X)

    print("Input shape:", X.shape)
    print("Output shape:", predictions.shape)
    print("Predictions:")
    print(predictions)