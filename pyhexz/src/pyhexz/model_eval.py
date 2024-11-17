"""Random scripts to analyze a model."""

from dataclasses import dataclass
import os
import numpy as np
import torch
from pyhexz.model import HexzNeuralNetwork
from pyhexz.modelrepo import LocalModelRepository


@dataclass
class WeightStats:
    # 5/25/50/75/95
    percentiles: list[float]
    mean: float
    std: float
    min: float
    max: float
    norm_l1: float
    norm_l2: float
    num_weights: int

    def iqr(self):
        return self.percentiles[3] - self.percentiles[1]

    def percentiles_str(self):
        return "[" + " ".join(f"{p:.6f}" for p in self.percentiles) + "]"


def weight_stats(model: torch.nn.Module):
    """Returns WeightStats for each relevant module type (Conv2d, Linear)."""
    weight_data = {
        "Conv2d": [],
        "Linear": [],
    }
    ignored_types = set()
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weight_data["Conv2d"].append(module.weight.detach().cpu().numpy())
        elif isinstance(module, torch.nn.Linear):
            weight_data["Linear"].append(module.weight.detach().cpu().numpy())
        else:
            t = type(module)
            ignored_types.add(f"{t.__module__}.{t.__name__}")
    # print("Ignored module types:", sorted(ignored_types))

    stats = {}
    for layer_type, weights in weight_data.items():
        flat_weights = np.concatenate([w.flatten() for w in weights])
        percentiles = np.percentile(flat_weights, [5, 25, 50, 75, 95])
        stats[layer_type] = WeightStats(
            percentiles=percentiles,
            mean=np.mean(flat_weights),
            std=np.std(flat_weights),
            min=np.min(flat_weights),
            max=np.max(flat_weights),
            norm_l1=torch.norm(torch.from_numpy(flat_weights), p=1).item(),
            norm_l2=torch.norm(torch.from_numpy(flat_weights), p=2).item(),
            num_weights=len(flat_weights),
        )
    return stats


def weights_timeline(model_name="res10", base_dir=None):
    if base_dir is None:
        base_dir = os.path.join(os.getenv("HOME"), "tmp/hexz-models")
    repo = LocalModelRepository(base_dir)
    latest = repo.get_latest_checkpoint(model_name)
    stats = []
    for i in range(latest + 1):
        model: HexzNeuralNetwork = repo.get_model(model_name, checkpoint=i)
        stats.append((i, weight_stats(model)))

    return stats
