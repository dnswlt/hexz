# Configuration used by the training server.
import os
import typing


def _from_env(cls):
    """Returns a config in which each field is optionally overridden by its corresponding
    environment variable HEXZ_FIELD_NAME.
    """
    envvars = {}
    for f in cls._fields:
        typ = cls.__annotations__.get(f)
        v = os.getenv(f"HEXZ_{f.upper()}")
        if v is not None:
            if typ == bool:
                v = v.lower() not in ['0', 'false', 'no', 'off', 'n', '']
            elif typ:
                v = typ(v)
            envvars[f] = v
    return cls(**envvars)


class TrainingConfig(typing.NamedTuple):
    model_repo_base_dir: str
    model_name: str
    # Minimum number of MCTS runs per move that a worker must have made
    # for its examples to be accepted. This prevents inadvertent bad training data.
    min_runs_per_move: int = 800
    # Model type. One of (conv2d, resnet). Only relevant if a new model is created at startup.
    model_type: str = "conv2d"
    model_blocks: int = 5
    # batch size to use for training
    batch_size: int = 4096
    # Train a new model after this many new examples were received:
    training_trigger_threshold: int = 100
    # Window size of (newest) examples to use when training a new model
    training_examples_window_size: int = 2**20
    # Training parameters
    num_epochs: int = 7
    learning_rate: float = 1e-3
    adam_weight_decay: float = 1e-4
    
    device: str = "cpu"
    shuffle: bool = True
    pin_memory: bool = False

    @classmethod
    def from_env(cls):
        return _from_env(cls)


class CPUEngineConfig(typing.NamedTuple):
    local_model_path: str

    @classmethod
    def from_env(cls):
        return _from_env(cls)
