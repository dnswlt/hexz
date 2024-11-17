"""Random scripts to analyze a model."""

import torch
from pyhexz.model import HexzNeuralNetwork
from pyhexz.modelrepo import LocalModelRepository


def weight_distrib(base_dir: str, model_name="res10"):
    """Loads the hexz model at path and prints a distribution of its weight values."""
    repo = LocalModelRepository(base_dir)
    model: HexzNeuralNetwork = repo.get_model(model_name)
    q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    for n, d in model.state_dict().items():
        if d.dtype != torch.float32:
            print(f"skipping {n} with dtype {d.dtype}")
            continue
        print(n, d.shape, torch.quantile(d.flatten(), q))


