"""Default Hyperparameters Shared by the CLI and the Public API."""

from __future__ import annotations

import ast
import configparser
from pathlib import Path

# Path to the configuration file shipped with the repository:
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.cfg"

# Fallbacks mirroring configs/config.cfg. They are only used when the file is
# unavailable (e.g., rl_ddpg installed without the configs directory):
_ENV_FALLBACK: dict[str, float] = {
    "min_ej": 0.05,
    "max_ej": 12.0,
    "min_ec": 0.15,
    "max_ec": 0.3,
    "min_ng": -1.0,
    "max_ng": 1.0,
}

_NET_FALLBACK: dict[str, list[int]] = {
    "layer_act_dims": [300, 300],
    "layer_crit_dims": [300, 300],
}

_AGENT_FALLBACK: dict[str, float | int] = {
    "lr_actor": 1e-4,
    "lr_critic": 3e-4,
    "rho": 0.005,
    "gamma": 0.98,
    "noise": 0.2,
    "buffer_size": 100000,
    "batch_size": 128,
}

_TRAIN_FALLBACK: dict[str, int] = {
    "seed": 32,
    "num_episodes": 100,
    "max_steps": 55,
}


def load_config(config_path: str | Path | None = None) -> configparser.ConfigParser:
    """Load configuration from a file.

    Args:
        config_path: Path to the configuration file.
            Defaults to <project_root>/configs/config.cfg.

    Returns:
        config: Configuration object.
    """

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    config.read(str(config_path if config_path is not None else CONFIG_PATH))
    return config


# Parsed once at import time, so that the default arguments of the environment,
# the agent, and the training loop match the values used by the CLI:
_DEFAULT_CONFIG = load_config()


def _resolve(config: configparser.ConfigParser | None) -> configparser.ConfigParser:
    """Return the given configuration object, or the default one."""

    return _DEFAULT_CONFIG if config is None else config


def _parse_dims(raw: str) -> list[int]:
    """Parse a hidden layer specification such as "[300, 300]"."""

    return [int(dim) for dim in ast.literal_eval(raw)]


def env_params(config: configparser.ConfigParser | None = None) -> dict[str, float]:
    """Read the environment hyperparameters from the [ENV] section.

    Args:
        config: Configuration object. Defaults to the shipped config file.

    Returns:
        Mapping of the observation space bounds.
    """

    config = _resolve(config)
    return {
        key: config.getfloat("ENV", key, fallback=fallback)
        for key, fallback in _ENV_FALLBACK.items()
    }


def net_params(config: configparser.ConfigParser | None = None) -> dict[str, list[int]]:
    """Read the neural network hyperparameters from the [NEURAL_NET] section.

    Args:
        config: Configuration object. Defaults to the shipped config file.

    Returns:
        Mapping of the actor and critic hidden layer sizes.
    """

    config = _resolve(config)
    return {
        key: _parse_dims(config.get("NEURAL_NET", key, fallback=str(fallback)))
        for key, fallback in _NET_FALLBACK.items()
    }


def agent_params(
    config: configparser.ConfigParser | None = None,
) -> dict[str, float | int]:
    """Read the agent hyperparameters from the [AGENT] section.

    Args:
        config: Configuration object. Defaults to the shipped config file.

    Returns:
        Mapping of the agent hyperparameters.
    """

    config = _resolve(config)
    return {
        key: (
            config.getint("AGENT", key, fallback=fallback)
            if isinstance(fallback, int)
            else config.getfloat("AGENT", key, fallback=fallback)
        )
        for key, fallback in _AGENT_FALLBACK.items()
    }


def train_params(config: configparser.ConfigParser | None = None) -> dict[str, int]:
    """Read the training hyperparameters from the [TRAIN] section.

    Args:
        config: Configuration object. Defaults to the shipped config file.

    Returns:
        Mapping of the training hyperparameters.
    """

    config = _resolve(config)
    return {
        key: config.getint("TRAIN", key, fallback=fallback)
        for key, fallback in _TRAIN_FALLBACK.items()
    }
