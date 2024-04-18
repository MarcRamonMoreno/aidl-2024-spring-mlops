import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    # You should build your model with at least 2 layers using tanh activation in between
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        # Apply the first layer and a tanh activation
        x = torch.tanh(self.layer1(x))
        # Output layer, no activation since it's a regression problem
        return self.layer2(x)
        