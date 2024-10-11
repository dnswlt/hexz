"""Python implementation of the Flagz game.

This file includes the neural network implementation.
"""

import torch
from torch import nn
import torch.nn.functional as F

from pyhexz.board import Board


class HexzCNNBlocks(nn.Module):
    cnn_filters = 128
    num_cnn_blocks = 5

    def __init__(self):
        super().__init__()
        self.cnn_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        Board.shape[0],
                        self.cnn_filters,
                        kernel_size=3,
                        bias=False,
                        padding="same",
                    ),  # [N, cnn_channels, 11, 10]
                    nn.BatchNorm2d(self.cnn_filters),
                    nn.ReLU(),
                )
            ]
            + [
                nn.Sequential(
                    nn.Conv2d(
                        self.cnn_filters,
                        self.cnn_filters,
                        kernel_size=3,
                        bias=False,
                        padding="same",
                    ),  # [cnn_channels, cnn_channels, 11, 10]
                    nn.BatchNorm2d(self.cnn_filters),
                    nn.ReLU(),
                )
                for i in range(self.num_cnn_blocks - 1)
            ]
        )

    def forward(self, x):
        for b in self.cnn_blocks:
            x = b(x)
        return x


class HexzNeuralNetwork(nn.Module):
    def __init__(self, traced=False):
        super().__init__()
        self.cnn_blocks = HexzCNNBlocks()
        if traced:
            input = torch.rand((1, *Board.shape))
            self.cnn_blocks = torch.jit.trace(self.cnn_blocks, input)
        self.policy_head = nn.Sequential(
            nn.Conv2d(HexzCNNBlocks.cnn_filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 11 * 10, 2 * 11 * 10),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(HexzCNNBlocks.cnn_filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(11 * 10, 11 * 10),
            nn.ReLU(),
            nn.Linear(11 * 10, 1),
        )

    def forward(self, x, action_mask):
        x = self.cnn_blocks(x)
        policy = self.policy_head(x)
        # Mask out (i.e. set to ~ 0 in the exp domain) all policy predictions for invalid actions.
        policy = policy.where(action_mask.flatten(1), torch.full_like(policy, -1e32))
        v = self.value_head(x)
        return F.log_softmax(policy, dim=1), torch.tanh(v)
