from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)