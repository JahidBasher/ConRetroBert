from functools import lru_cache
from typing import List, Optional, Tuple

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


ReactantSet = Tuple[str, ...]


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import rdChemReactions
    except Exception as exc:
        raise ImportError("RDKit is required for chemistry utilities.") from exc
    return Chem, rdChemReactions


@lru_cache(maxsize=100_000)
def _canon_smiles(smiles: str) -> Optional[str]:
    Chem, _ = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


@lru_cache(maxsize=50_000)
def _rxn_from_smarts(template_smarts: str):
    _, rdChemReactions = _require_rdkit()
    try:
        return rdChemReactions.ReactionFromSmarts(template_smarts)
    except Exception:
        return None


@lru_cache(maxsize=50_000)
def _reverse_rxn_from_smarts(template_smarts: str):
    parts = template_smarts.split(">>")
    if len(parts) != 2:
        return None
    forward_smarts = parts[1] + ">>" + parts[0]
    return _rxn_from_smarts(forward_smarts)


def apply_template(product_smiles: str, template_smarts: str) -> List[ReactantSet]:
    Chem, _ = _require_rdkit()
    product_mol = Chem.MolFromSmiles(product_smiles)
    if product_mol is None:
        return []

    rxn = _rxn_from_smarts(template_smarts)
    if rxn is None:
        return []

    try:
        reactant_sets = rxn.RunReactants((product_mol,))
    except Exception:
        return []

    outcomes: List[ReactantSet] = []
    for reactants in reactant_sets:
        smiles_list: List[str] = []
        valid = True
        for reactant in reactants:
            if reactant is None:
                valid = False
                break
            try:
                smi = Chem.MolToSmiles(reactant, canonical=True)
            except Exception:
                valid = False
                break
            if not smi:
                valid = False
                break
            smiles_list.append(smi)
        if valid and smiles_list:
            outcomes.append(tuple(smiles_list))
    return outcomes


def forward_check(product_smiles: str, template_smarts: str, reactants: ReactantSet) -> bool:
    Chem, _ = _require_rdkit()
    rxn = _reverse_rxn_from_smarts(template_smarts)
    if rxn is None:
        return False

    reactant_mols = []
    for smi in reactants:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        reactant_mols.append(mol)

    try:
        product_sets = rxn.RunReactants(tuple(reactant_mols))
    except Exception:
        return False

    product_canon = _canon_smiles(product_smiles)
    if product_canon is None:
        return False

    for products in product_sets:
        for prod in products:
            if prod is None:
                continue
            try:
                smi = Chem.MolToSmiles(prod, canonical=True)
            except Exception:
                continue
            if smi == product_canon:
                return True
    return False

