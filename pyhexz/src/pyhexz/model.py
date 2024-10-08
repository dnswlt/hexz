"""Python implementation of the Flagz game.

This file includes the neural network implementation.
"""

from bisect import bisect_left
import glob
import h5py
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F


class HexzNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        board_channels = 11
        cnn_filters = 128
        num_cnn_blocks = 5
        self.cnn_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        board_channels,
                        cnn_filters,
                        kernel_size=3,
                        bias=False,
                        padding="same",
                    ),  # [N, cnn_channels, 11, 10]
                    nn.BatchNorm2d(cnn_filters),
                    nn.ReLU(),
                )
            ]
            + [
                nn.Sequential(
                    nn.Conv2d(
                        cnn_filters,
                        cnn_filters,
                        kernel_size=3,
                        bias=False,
                        padding="same",
                    ),  # [cnn_channels, cnn_channels, 11, 10]
                    nn.BatchNorm2d(cnn_filters),
                    nn.ReLU(),
                )
                for i in range(num_cnn_blocks - 1)
            ]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(cnn_filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 11 * 10, 2 * 11 * 10),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(cnn_filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(11 * 10, 11 * 10),
            nn.ReLU(),
            nn.Linear(11 * 10, 1),
        )

    def forward(self, x, action_mask):
        for b in self.cnn_blocks:
            x = b(x)
        policy = self.policy_head(x)
        # Mask out (i.e. set to ~ 0 in the exp domain) all policy predictions for invalid actions.
        policy = policy.where(action_mask.flatten(1), torch.full_like(policy, -1e32))
        v = self.value_head(x)
        return F.log_softmax(policy, dim=1), torch.tanh(v)
