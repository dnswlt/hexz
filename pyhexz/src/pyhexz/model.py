"""This file contains the neural network implementations."""

import torch
from torch import nn

from pyhexz.board import Board


class RichBoardFeatures(nn.Module):
    """Expand the canonical board tensor into explicit, learning-friendly planes.

    The canonical 11-plane representation is intentionally kept as the storage
    and inference API.  This module derives categorical and global features
    inside the model, so old replay and C++ clients remain compatible.
    """

    channels = 45

    def __init__(self):
        super().__init__()
        valid = torch.ones((1, 1, 11, 10), dtype=torch.float32)
        valid[:, :, 1::2, 9] = 0
        parity = torch.zeros((1, 1, 11, 10), dtype=torch.float32)
        parity[:, :, 1::2, :] = 1
        self.register_buffer("valid_cells", valid)
        self.register_buffer("row_parity", parity)

    def forward(
        self, board: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        batch = board.shape[0]
        features = [
            self.valid_cells.expand(batch, -1, -1, -1),
            self.row_parity.expand(batch, -1, -1, -1),
            board[:, 0:1],  # current-player flags
            board[:, 5:6],  # opponent flags
        ]

        # Occupied values are categorical game states, not continuous
        # quantities.  Make that distinction explicit for both players.
        for channel in (1, 6):
            values = board[:, channel : channel + 1]
            for value in range(1, 6):
                features.append((values == value).to(board.dtype))

        features.extend((board[:, 2:3], board[:, 7:8]))

        # The propagated next value has the same categorical semantics.
        for channel in (3, 8):
            values = board[:, channel : channel + 1]
            for value in range(1, 6):
                features.append((values == value).to(board.dtype))

        # Remaining flags are global state. They are stored as constant-valued
        # planes; use an explicit one-hot encoding for the legal range 0..3.
        for channel in (4, 9):
            remaining = board[:, channel : channel + 1]
            for value in range(4):
                features.append((remaining == value).to(board.dtype))

        grass = board[:, 10:11]
        for value in range(1, 6):
            features.append((grass == value).to(board.dtype))

        # Legal actions are strategically useful inputs in addition to being
        # the final policy mask.
        features.extend((action_mask[:, 0:1].to(board.dtype),
                         action_mask[:, 1:2].to(board.dtype)))

        # Make the global race explicit. The denominator is the maximum score
        # on the 105-cell board and safely bounds all reachable scores.
        own_score = board[:, 1:2].sum(dim=(2, 3), keepdim=True) / 525.0
        opponent_score = board[:, 6:7].sum(dim=(2, 3), keepdim=True) / 525.0
        score_diff = own_score - opponent_score
        features.extend(
            (
                own_score.expand(-1, -1, 11, 10),
                opponent_score.expand(-1, -1, 11, 10),
                score_diff.expand(-1, -1, 11, 10),
            )
        )

        occupied = (
            (board[:, 0:1] != 0).to(board.dtype)
            + (board[:, 1:2] != 0).to(board.dtype)
            + (board[:, 5:6] != 0).to(board.dtype)
            + (board[:, 6:7] != 0).to(board.dtype)
        )
        phase = occupied.sum(dim=(2, 3), keepdim=True) / 105.0
        features.append(phase.expand(-1, -1, 11, 10))
        return torch.cat(features, dim=1)


class CNNLayer(nn.Module):
    """CNNLayer is a CNN-based torso of the alpha zero style model.
    It consists of `blocks` many CNN "blocks", which themselves
    consist of a Conv2d, a BatchNorm2d, and a ReLU layer.

    The number of blocks and Conv2d filters and the kernel size
    can be adjusted via __init__ parameters.
    """

    def __init__(
        self, blocks=5, filters=128, kernel_size=3, input_channels=Board.shape[0]
    ):
        super().__init__()
        self._blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        input_channels,
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

    def __init__(
        self, blocks=5, filters=128, kernel_size=3, input_channels=Board.shape[0]
    ):
        super().__init__()
        self._blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        input_channels,
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

    def __init__(
        self,
        blocks=5,
        filters=128,
        model_type="conv2d",
        representation="legacy",
    ):
        super().__init__()
        if representation not in ("legacy", "rich_v1"):
            raise ValueError(f"Invalid representation: {representation}")
        # Save parameters of the network, so they can be saved together
        # with the model's state dict.
        self.ctor_args = dict(
            blocks=blocks,
            filters=filters,
            model_type=model_type,
            representation=representation,
        )
        self.representation = representation
        if representation == "rich_v1":
            self.feature_encoder = RichBoardFeatures()
            input_channels = RichBoardFeatures.channels
        else:
            self.feature_encoder = None
            input_channels = Board.shape[0]
        if model_type == "conv2d":
            self._torso = CNNLayer(
                blocks=blocks, filters=filters, input_channels=input_channels
            )
        elif model_type == "resnet":
            self._torso = ResidualLayer(
                blocks=blocks, filters=filters, input_channels=input_channels
            )
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 11 * 10, 2 * 11 * 10),
        )
        if representation == "rich_v1":
            self.value_head = nn.Sequential(
                nn.Conv2d(filters, 32, kernel_size=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 11 * 10, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        else:
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
        if self.feature_encoder is not None:
            b = self.feature_encoder(b, action_mask)
        x = self._torso(b)
        policy = self.policy_head(x)
        # Mask out (i.e. set to ~ 0 in the exp domain) all policy predictions for invalid actions.
        policy = policy.where(action_mask.flatten(1), torch.full_like(policy, -1e32))
        v = self.value_head(x)
        return policy, torch.tanh(v)
