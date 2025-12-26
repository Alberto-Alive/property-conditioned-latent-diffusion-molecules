from __future__ import annotations
import argparse
import os
import random
from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm

from moldifflite.chemistry.selfies_utils import smiles_to_selfies, split_selfies
from moldifflite.chemistry.rdkit_props import DEFAULT_PROP_NAMES, canonicalize_smiles, compute_properties
from moldifflite.utils import save_json