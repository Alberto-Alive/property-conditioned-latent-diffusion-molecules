from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from .rdkit_props import canonicalize_smiles, compute_properties

def compute_basic_metrics(gen_smiles: List[str], train_smiles_set: Optional[Set[str]] = None) -> Dict[str, float]:
    canon = [
        
    ]
    valid_flags = [
        
    ]
    
    for s in gen_smiles:
        c = canonicalize_smiles(s) if s is not None else None
        if c is None:
            valid_flags.append(False)
        else:
            valid_flags.append(True)
            canon.append(c)
    validity = float(np.mean(valid_flags)) if len(valid_flags) > 0 else 0.0
    unique = len(set(canon)) / max(len(canon), 1)
    
    novelty = None
    if train_smiles_set is not None:
        novelty = len([s for s in set(canon) if s not in train_smiles_set]) / max(len(set(canon)), 1)
    out = {"validity": validity, "uniqueness": float(unique)}
    if novelty is not None:
        out["novelty"] = float(novelty)
    return out

def satisfaction_rate(
    smiles_list: List[str],
    constraints: Dict[str, Tuple[float, float]],
    prop_names: List[str],
) -> float:
    ok = 0
    tot = 0
    for s in smiles_list:
        if s is None:
            continue
        props = compute_properties(s, prop_names=prop_names)
        if props is None:
            continue
        tot +=1 
        good = True
        for p, (lo, hi) in constraints.items():
            v = float(props[p])
            if v < lo or v > hi:
                good = False
                break
        if good:
            ok += 1
    return float(ok / max(tot, 1))