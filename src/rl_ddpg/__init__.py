# -*- coding: utf-8 -*-

# This code is part of rl_ddpg.
#
# (C) Copyright Lucas Camponogara Viera, 2026.
#
# This code is licensed under the Apache 2.0 License.
# You may obtain a copy of the License in the root directory of this source tree.

"""rl_ddpg 2026"""

# Check for required libraries:
import rl_ddpg.sanity as sanity  # noqa: F401

# Standard library imports:
import sys
from pathlib import Path

# Third-party imports:
import numpy as np  # For simple numerical operations.
import scqubits  # For quantum computing.
import tensorflow as tf  # For tensor operations and deep learning.

# Local imports:
from rl_ddpg.envs import TransmonQubitEnv  # noqa: F401
from rl_ddpg.rl import DDPGAgent, train  # noqa: F401
from rl_ddpg.utils import set_global_seed  # noqa: F401

__all__ = ["DDPGAgent", "TransmonQubitEnv", "about", "set_global_seed", "train"]

# Package Info:
VERSION_PATH = Path(__file__).resolve().with_name("VERSION.txt")
VERSION = VERSION_PATH.read_text(encoding="utf-8").strip()

__name__ = "rl_ddpg"
__version__ = VERSION
__status__ = "Development"
__homepage__ = "https://github.com/camponogaraviera/rl-ddpg"
__author__ = "Lucas Camponogara Viera"
__license__ = "Apache 2.0"
__copyright__ = "Copyright LCV 2026"


def about():
    """Function to display the rl_ddpg project information."""
    print(" \
    ###################################\n \
    RL DDPG Information:\n \
    ###################################\n")
    print(f"{__copyright__}")
    print(f"Name: {__name__}")
    print(f"Version: {__version__}")
    print(f"Status: {__status__}")
    print(f"Home-page: {__homepage__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print(
        f"Requires: python=="
        f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}",
        f"numpy=={np.__version__}",
        f"scqubits=={scqubits.__version__}",
        f"tensorflow=={tf.__version__}",
    )
