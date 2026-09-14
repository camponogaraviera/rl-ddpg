"""Process-Wide Seeding for Reproducibility."""

import random

import numpy as np
import tensorflow as tf

from rl_ddpg.utils.config import train_params

# Default read from configs/config.cfg, so that seeding without an explicit value
# matches the seed used by the `rl_ddpg` command line:
_TRAIN_DEFAULTS = train_params()


def set_global_seed(seed: int = _TRAIN_DEFAULTS["seed"]) -> None:
    """Set process-wide seeds for reproducible training and evaluation.

    Call it before instantiating the agent, so that the network weight
    initialization is reproducible as well.

    Args:
        seed: Seed shared by Python, NumPy, and TensorFlow.
            Defaults to the seed in configs/config.cfg.

    Returns:
        None.
    """

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
