"""Python implementation of the Flagz game.

This file includes the neural network implementation.
"""

import torch
from torch import nn

from pyhexz.board import Board


class CNNBlocks(nn.Module):

    def __init__(self, blocks=5, filters=128, kernel_size=3):
        super().__init__()
        self._blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        Board.shape[0],
                        filters,
                        kernel_size=kernel_size,
                        bias=False,
                        padding="same",
                    ),  # [N, cnn_channels, 11, 10]
                    nn.BatchNorm2d(filters),
                    nn.ReLU(),
                )
            ]
            + [
                nn.Sequential(
                    nn.Conv2d(
                        filters,
                        filters,
                        kernel_size=kernel_size,
                        bias=False,
                        padding="same",
                    ),  # [cnn_channels, cnn_channels, 11, 10]
                    nn.BatchNorm2d(filters),
                    nn.ReLU(),
                )
                for _ in range(blocks - 1)
            ]
        )

    def forward(self, x):
        for b in self._blocks:
            x = b(x)
        return x


class HexzNeuralNetwork(nn.Module):
    def __init__(self, cnn_blocks=5, cnn_filters=128):
        super().__init__()
        self.cnn_blocks = CNNBlocks(blocks=cnn_blocks, filters=cnn_filters)
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

    def forward(self, b: torch.Tensor, action_mask: torch.Tensor):
        """
        Arguments:
            b: a batch of hexz boards of shape (11, 11, 10).
            action_mask: a batch of action masks of shape (2, 11, 10).

        Returns:
            A tuple of (policy, value) tensors. 
            
            policy is of shape (N, 2 * 11 * 10) and contains the *raw logits* of the move policy.
            During inference, clients should call softmax on the logits to get the policy's move
            likelihoods. They probably also want to .reshape(-1, 2, 11, 10) the output
            to get the move likelihoods per piece (0=flag, 1=normal) and board cell.
            
            The value tensor is of shape (N, 1) and contains the value of the input board.
            1 means the board looks like a clear win from the perspective of the current player,
            -1 predicts a clear loss, and 0 is a draw.
        """
        x = self.cnn_blocks(b)
        policy = self.policy_head(x)
        # Mask out (i.e. set to ~ 0 in the exp domain) all policy predictions for invalid actions.
        policy = policy.where(action_mask.flatten(1), torch.full_like(policy, -1e32))
        v = self.value_head(x)
        return policy, torch.tanh(v)
