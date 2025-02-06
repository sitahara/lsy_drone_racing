import torch
import torch.nn as nn


class ResidualPytorchModel(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_dim=128):
        super(ResidualPytorchModel, self).__init__()
        self.input_dim = state_dim + control_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        # Block 1
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(p=0.2)

        # Block 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(p=0.2)

        # Block 3
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(p=0.2)

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, state_dim)

        # Initialize weights
        self._initialize_weights()

    def forward(self, state, control):
        x = torch.cat((state, control), dim=-1)

        # Block 1
        residual = x
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x += residual  # Skip connection

        # Block 2
        residual = x
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x += residual  # Skip connection

        # Block 3
        residual = x
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x += residual  # Skip connection

        # Output layer
        residual = self.fc_out(x)
        return residual

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
