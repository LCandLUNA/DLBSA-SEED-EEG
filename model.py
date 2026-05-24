# models.py

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Empty model template.

    You must:
    - define layers in __init__
    - implement forward pass

    Input shape:  (B, C, T)
    Output shape: (B, num_classes) for classification
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        in_channels = config["dataset"]["input_channels"]
        num_classes = config["dataset"]["num_classes"]

        # -------------------------
        # TODO: Define your layers here
        # Example:
        # self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3)
        # self.fc = nn.Linear(..., num_classes)
        # -------------------------

        pass

    def forward(self, x):
        """
        Forward pass

        x: (B, C, T)
        """

        # -------------------------
        # TODO: Implement forward pass
        # Example:
        # x = self.conv1(x)
        # x = ...
        # return x
        # -------------------------

        raise NotImplementedError("You must implement the forward pass.")