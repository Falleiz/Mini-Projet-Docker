import torch
import torch.nn as nn


class DNNRegressor(nn.Module):
    """
    Deep Neural Network for Regression.
    Architecture: Input -> [Hidden Layers] -> Output
    """

    def __init__(self, input_size, hidden_sizes=[320, 160, 80, 40], dropout_rate=0.28):
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()
