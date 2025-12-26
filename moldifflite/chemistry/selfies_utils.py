from __future__ import annotations
from typing import List, Optional
import selfies as sf


def smiles_to_selfies(smiles: str) -> Optional[str]:
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


def selfies_to_smiles(selfies: str) -> Optional[str]:
    try:
        return sf.decoder(selfies)
    except Exception:
        return None


def split_selfies(selfies: str) -> List[str]:
    return list(sf.split_selfies(selfies))


def join_selfies(tokens: List[str]) -> str:
    return "".join(tokens)
