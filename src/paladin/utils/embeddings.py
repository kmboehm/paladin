import os
import json
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig


def load_if_exists(emb_cfg: DictConfig, mandatory_keys: Optional[List[str]] = None) -> Optional[Dict]:
    """Load a json embedding mapping if the path exists.

    Args:
        emb_cfg (DictConfig): The embedding configuration with keys path and dim.
        mandatory_keys (List[str]): The mandatory keys in the mapping.

    Returns:
        Optional[Dict]: The embedding mapping if it exists, else None.
    """
    path = emb_cfg.get("path")
    emb_dim = emb_cfg.get("dim")

    if path is None or not os.path.exists(path):
        print ("Skipping embedding...")
        return None

    print(f"Loading embedding from {path}")
    with open(path, "r") as f:
        mapping = json.load(f)

    for target, emb in mapping.items():
        assert len(emb) == emb_dim, f"The embedding for {target} must have dimension {emb_dim}."
    # for key in [
    #     "fraction_genome_altered",
    #     "impact_tmb_score",
    #     "msi_type",
    #     "COAD-msi_type.0",
    #     "COAD-msi_type.1",
    #     "UEC.msi_type.0",
    #     "UEC.msi_type.1",
    #     "IDC-HR.1-HER2.1",
    #     "IDC-HR.1-HER2.0",
    #     "IDC-HR.0-HER2.1",
    #     "IDC-HR.0-HER2.0",
    #     "UEC-msi_type.0",
    #     "HGSOC",
    #     "UEC-msi_type.1",
    #     "BRCANOS",
    # ]:
    #     if key not in mapping:
    #         mapping[key] = torch.zeros(emb_dim)
    for key_missing, closest_key in [
        ("ECAD", "CEAD"),
        ("HGSOC", "SOC"),
        ("BRCANOS", "BRCA"),
        ("URCC", "RCC"),
        ("PDC", "CUP"),
        ("EHCH", "CHOL"),
        ("NSCLCPD", "NSCLC"),
        ("HGGNOS", "GBM"),
        ("WDLS", "DDLS"),
        ("CUPNOS", "CUP"),
        ("SPDAC", "USTAD"),
        ("HGSFT", "SOC"),
        ("UMEC", "UCEC"),
    ]:
        if key_missing not in mapping:
            try:
                mapping[key_missing] = mapping[closest_key]
            except KeyError:
                print(f"Warning: {key_missing} not found in embedding mapping, returning zero embedding")
                mapping[key_missing] = torch.zeros(emb_dim)

    if mandatory_keys is not None:
        for key in mandatory_keys:
            if key not in mapping:
                if "pathway" in key.lower():
                    mapping[key] = torch.zeros(emb_dim)
                if "+" in key:
                    individual_keys = key.split("+")
                    sum_emb = torch.zeros(emb_dim)
                    for individual_key in individual_keys:
                        sum_emb += torch.tensor(mapping[individual_key])
                    mapping[key] = sum_emb
                else:
                    print(f"Warning: {key} not found in embedding mapping, returning zero embedding")
                    mapping[key] = torch.zeros(emb_dim)

    tensor_dict = {k: torch.tensor(v) for k, v in mapping.items()}

    return tensor_dict
