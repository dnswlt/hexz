
# Configuration used by the training server.
import collections
import os
import typing


def _from_env(cls):
    envvars = {}
    for f in cls._fields:
        typ = cls.__annotations__.get(f)
        v = os.getenv(f"HEXZ_{f.upper()}")
        if v is not None:
            if typ:
                v = typ(v)
            envvars[f] = v
    return cls(**envvars)


class TrainingConfig(typing.NamedTuple):
    model_repo_base_dir: str
    model_name: str
    batch_size: int = 4096
    num_epochs: int = 7
    device: str = "cpu"
    shuffle: bool = True
    pin_memory: bool = False

    @classmethod
    def from_env(cls):
        """Returns a config in which each field is optionally overridden by its corresponding
         environment variable HEXZ_FIELD_NAME.
        """
        return _from_env(cls)
