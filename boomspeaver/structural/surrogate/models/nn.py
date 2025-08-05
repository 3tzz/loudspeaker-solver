import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(
        self, input_dim, hidden_dims, output_dim, activation="relu", dropout=0.0
    ):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "gelu":
                    layers.append(nn.GELU())

                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
