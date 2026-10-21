"""Seeding and environment capture.

Seeds make runs repeatable on the same machine and library versions. They do
not guarantee bit-identical results across hardware, library versions or
non-deterministic GPU kernels.
"""

from __future__ import annotations

import platform
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def environment_info() -> dict:
    import opacus
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "opacus": opacus.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
    }
