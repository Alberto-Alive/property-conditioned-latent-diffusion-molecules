from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

DEFAULT_PROP_NAMES = ["qed", "logp", "mw", "tpsa", "hbd"]

def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit.Chem import QED
        from rdkit.Chem import Crippen
        from rdkit.Chem import rdMolDescriptors
    except Exception as e:
        raise ImportError(
            "RDKit is required for property computation. Install via conda-forge:\n"
            "  conda install -c conda-forge rdkit"
        ) from e
        
def canonicalize_smiles(smiles:str) -> Optional[str]:
    _require_rdkit()
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def smiles_is_valid(smiles: str) -> bool:
    _require_rdkit()
    from rdkit import Chem
    return Chem.MolFromSmiles(smiles) is not None
    