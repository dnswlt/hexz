"""This file contains the neural network implementations."""

import torch
from torch import nn

from pyhexz.board import Board


class CNNLayer(nn.Module):
    """CNNLayer is a CNN-based torso of the alpha zero style model.
    It consists of `blocks` many CNN "blocks", which themselves
    consist of a Conv2d, a BatchNorm2d, and a ReLU layer.

    The number of blocks and Conv2d filters and the kernel size
    can be adjusted via __init__ parameters.
    """

    def __init__(self, blocks=5, filters=128, kernel_size=3):
        super().__init__()
        self._blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        Board.shape[0],
                        filters,
                        kernel_size=kernel_size,
                        # No bias, it would be redundant as a BatchNorm2d layer follows immediately.
                        bias=False,
                        padding="same",
                    ),  # [N, filters, 11, 10]
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
                    ),  # [filters, filters, 11, 10]
                    nn.BatchNorm2d(filters),
                    nn.ReLU(),
                )
                for _ in range(blocks - 1)
            ]
        )

    def forward(self, x):
        """
        Arguments:
            (N, 11, 11, 10) tensor batches representing hexz boards.
        
        Returns:
            (N, `filters`, 11, 10) CNN outputs
        """
        for b in self._blocks:
            x = b(x)
        return x


class ResidualBlock(nn.Module):
    """ResidualBlock is a single component of a ResidualLayer.

    A residual layer would typically contain 5-20 blocks in sequence.
    """

    def __init__(self, filters=128, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            filters, filters, kernel_size=kernel_size, padding="same", bias=False
        )
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            filters, filters, kernel_size=kernel_size, padding="same", bias=False
        )
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x


class ResidualLayer(nn.Module):

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
                    ),  # [N, filters, 11, 10]
                    nn.BatchNorm2d(filters),
                    nn.ReLU(),
                )
            ]
            + [
                ResidualBlock(filters=filters, kernel_size=kernel_size)
                for _ in range(blocks - 1)
            ]
        )

    def forward(self, x):
        for b in self._blocks:
            x = b(x)
        return x


class HexzNeuralNetwork(nn.Module):
    def __init__(self, blocks=5, filters=128, model_type="conv2d"):
        super().__init__()
        if model_type == "conv2d":
            self._torso = CNNLayer(blocks=blocks, filters=filters)
        elif model_type == "resnet":
            self._torso = ResidualLayer(blocks=blocks, filters=filters)
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 11 * 10, 2 * 11 * 10),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, 1, kernel_size=1, bias=False),
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
            b: a batch of hexz boards of shape (N, 11, 11, 10).
            action_mask: a batch of action masks of shape (N, 2, 11, 10).

        Returns:
            A tuple of (policy, value) tensors.

            policy is of shape (N, 2 * 11 * 10) and contains the *raw logits* of the move policy.
            During inference, clients should call softmax on the logits to get the policy's move
            likelihoods. They probably also want to .reshape(-1, 2, 11, 10) the output
            to get the move likelihoods per piece (0=flag, 1=normal) and board cell.

            The value tensor is of shape (N, 1) and contains the predicted value of the input board.
            Values close to 1 predict a win for the current player,
            -1 predicts a clear loss, and 0 is a draw.
        """
        x = self._torso(b)
        policy = self.policy_head(x)
        # Mask out (i.e. set to ~ 0 in the exp domain) all policy predictions for invalid actions.
        policy = policy.where(action_mask.flatten(1), torch.full_like(policy, -1e32))
        v = self.value_head(x)
        return policy, torch.tanh(v)
