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

def compute_properties(smiles: str, prop_names: List[str] = DEFAULT_PROP_NAMES) -> Optional[Dict[str, float]]:
    _require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import QED, Crippen, Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    props: Dict[str, float] = {}
    for p in prop_names:
        if p == "qed":
            props[p] = float(QED.qed(mol))
        elif p == "logp":
            props[p] = float(Crippen.MolLogP(mol))
        elif p == "mw":
            props[p] = float(Descriptors.MolWt(mol))
        elif p == "tpsa":
            props[p] = float(rdMolDescriptors.CalcTPSA(mol))
        elif p == "hbd":
            props[p] = float(rdMolDescriptors.CalcNumHBD(mol))
        elif p == "hba":
            props[p] = float(rdMolDescriptors.CalcNumHBA(mol))
        elif p == "rotb":
            props[p] = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
        else:
            raise ValueError(f"Unknown property: {p}")
    return props