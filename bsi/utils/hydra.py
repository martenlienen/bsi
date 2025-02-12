import shlex

import hydra


def load_config(overrides: str | list[str] | None = None):
    """Load a config for train.py with the given command line overrides."""

    if overrides is None:
        overrides = []
    elif isinstance(overrides, str):
        overrides = shlex.split(overrides)

    with hydra.initialize(config_path="../config", version_base=None):
        return hydra.compose(config_name="train", overrides=overrides)
