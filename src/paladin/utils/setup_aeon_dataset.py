import json
import logging
import os
from typing import Dict, List, Tuple

import hydra
import omegaconf
import pandas as pd
import torch
from torch.utils.data import Dataset

from nn_core.common import PROJECT_ROOT

from paladin.utils.data_prep import create_joint_dataset, prepare_run_specific_frame
from paladin.utils.data_validation import assert_no_dataleak, assert_no_duplicates, validate_embeddings
from paladin.utils.embeddings import load_if_exists

pylogger = logging.getLogger(__name__)

histologies_to_move_to_underspecified = [
    "UDMN",
    "ADNOS",
    "CUP",
    "CUPNOS",
    "BRCNOS",
    "GNOS",
    "SCCNOS",
    "PDC",
    "NSCLC",
    "BRCA",
    "SARCNOS",
    "NETNOS",
    "MEL",
    "RCC",
    "BRCANOS",
    "COADREAD",
    "MUP",
    "NECNOS",
    "UCEC",
    "NOT",
]
underspec_subtypes = ["BRCNOS", "GNOS", "NSCLC", "BRCA", "SARCNOS", "MEL", "RCC", "BRCANOS", "COADREAD", "MUP", "UCEC"]
underspec_subtypes2 = ["CHOL", "APAD", "SBC", "GINET", "GBC", "AMPCA", "DIFG"]
cup_subtypes = ["UDMN", "CUP", "ADNOS", "CUPNOS", "SCCNOS", "NETNOS", "NECNOS", "PDC", "NVRINT", "NOT", "SCUP", "SPDAC"]
mixed_subtypes = ["MDLC", "CSCLC", "NSCLCPD", "UMEC", "PAASC", "MXOV", "URCC", "LUAS"]
histologies_to_move_to_underspecified = (
    histologies_to_move_to_underspecified + underspec_subtypes + underspec_subtypes2 + cup_subtypes + mixed_subtypes
)
histologies_to_move_to_underspecified = list(set(histologies_to_move_to_underspecified))
del underspec_subtypes, underspec_subtypes2, cup_subtypes, mixed_subtypes
# print('\n'.join(f"  - {x}" for x in histologies_to_move_to_underspecified))


def create_target_mapping(
    unique_targets: List[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    # create one-hot encoding for targets
    target_mapping = {}
    target_to_int_mapping = {}
    for i, target_name in enumerate(unique_targets):
        one_hot_tensor = torch.zeros(len(unique_targets))
        one_hot_tensor[i] = 1
        target_mapping[target_name] = one_hot_tensor
        target_to_int_mapping[target_name] = i

    return target_mapping, target_to_int_mapping, unique_targets


def get_ncit_smoothed_target_mapping(
    neighbor_mapping_path: str, target_to_int_mapping: Dict[str, int]
) -> Dict[str, torch.Tensor]:
    # Load the target mapping
    with open(neighbor_mapping_path, "r") as f:
        neighbor_mapping = json.load(f)

    final_weights = {}
    smoothing_factor = 0.05
    # For each target in our target mapping
    for target_name in target_to_int_mapping.keys():
        if target_name not in neighbor_mapping:
            neighbor_weights = {}
        else:
            neighbor_weights = neighbor_mapping[target_name]

        # Create tensor for neighbors
        neighbor_tensor = torch.zeros(len(target_to_int_mapping))
        for neighbor, weight in neighbor_weights.items():
            if neighbor not in target_to_int_mapping:
                continue
            neighbor_tensor[target_to_int_mapping[neighbor]] = weight

        # Normalize the tensor
        if torch.sum(neighbor_tensor) > 0:
            neighbor_tensor = neighbor_tensor / torch.sum(neighbor_tensor)

        vanilla_crossentropy_weight = torch.zeros(len(target_to_int_mapping))
        vanilla_crossentropy_weight[target_to_int_mapping[target_name]] = 1.0

        final_weights[target_name] = (
            vanilla_crossentropy_weight * (1 - smoothing_factor) + neighbor_tensor * smoothing_factor
        )

    return final_weights


def make_one_hot(x, num_classes):
    one_hot = torch.zeros(num_classes)
    one_hot[x] = 1
    return one_hot


def load_aeon_dataset(
    sample_df_path: str,
    tasks: List[Dict],
    tile_tensor_url_col: str,
    tile_emb_dim: int,
    max_seq_len: int,
    histologic_emb: Dict,
    do_ontology_smoothing: bool = True,
    **kwargs,
) -> Dict[str, Dataset]:
    """Load the datasets.

    Args:
        sample_df_path (str): The path to the sample dataframe in parquet format with image_id, sample_id, {tasks}, and {tile_tensor_url_col} columns.
        targets (List[Dict): The targets, as {target: str, task: str, target_type: str, site: str/list} dictionaries.
        tile_tensor_url (str): The column name with the URL to the tile tensor files.
        tile_emb_dim (int): The tile embedding dimension.
        max_seq_len (int): The maximum sequence length.
        histologic_emb (Dict): The histologic embedding mapping path and dimension.
        **kwargs: Additional keyword arguments.

    Returns:
        Dict[str, Dataset]: The datasets for train, val, and test.
    """
    # load and validate data
    sample_df, _ = prepare_run_specific_frame(
        sample_df_path, tile_tensor_url_col, tasks, limit_histologies=False, drop_vus=False
    )

    assert "tcga" in sample_df["split"].unique(), "tcga split not in sample_df"

    sample_df["SEX"] = sample_df["SEX"].fillna("Unknown")
    sex_map_path = os.environ.get("AEON_SEX_MAP_PATH", "/gpfs/mskmind_ess/boehmk/data-commons/freeze/sex_original_to_idx.csv")
    if not os.path.exists(sex_map_path):
        pylogger.warning(f"Sex mapping file not found at {sex_map_path}. Using default index 2 for all samples.")
        sample_df["SEX"] = 2
    else:
        sex_map = pd.read_csv(sex_map_path)  # columns: idx, SEX
        sample_df["SEX"] = sample_df["SEX"].map(sex_map.set_index("SEX")["idx"]).fillna(2).astype(int)

    sample_df["TISSUE_SITE"] = sample_df["METASTATIC_SITE_x"].fillna("Unknown")
    sample_df.loc[sample_df.site == "Primary", "TISSUE_SITE"] = sample_df.loc[
        sample_df.site == "Primary", "PRIMARY_SITE_y"
    ].fillna("Unknown")
    tissue_site_map_path = os.environ.get("AEON_TISSUE_SITE_MAP_PATH", "/gpfs/mskmind_ess/boehmk/data-commons/freeze/tissue_site_original_to_idx.csv")
    if not os.path.exists(tissue_site_map_path):
        pylogger.warning(f"Tissue site mapping file not found at {tissue_site_map_path}. Using default index 8 for all samples.")
        sample_df["TISSUE_SITE"] = 8
    else:
        tissue_site_map = pd.read_csv(tissue_site_map_path)  # columns: idx, TISSUE_SITE
        sample_df["TISSUE_SITE"] = (
            sample_df["TISSUE_SITE"].map(tissue_site_map.set_index("TISSUE_SITE")["idx"]).fillna(8).astype(int)
        )
    # convert to torch tensor one-hot encoding
    sample_df["SEX"] = sample_df["SEX"].apply(lambda x: make_one_hot(x, 3))
    sample_df["TISSUE_SITE"] = sample_df["TISSUE_SITE"].apply(lambda x: make_one_hot(x, 57))

    for histology in histologies_to_move_to_underspecified:
        sample_df.loc[sample_df["oncotree_code_aeon"] == histology, "split"] = "underspecified"

    print(sample_df[sample_df["split"] == "underspecified"]["oncotree_code_aeon"].value_counts())
    # exit()

    # possibly aggregate histologies
    mapping = tasks[0]["oncotree_mapping"]
    for k, v in mapping.items():
        sample_df.loc[sample_df["oncotree_code_aeon"] == k, "oncotree_code_aeon"] = v

    sample_df = sample_df[
        sample_df["oncotree_code_aeon"].isin(tasks[0]["histologies"])
    ]  # limit to histologies only after mapping

    target_mapping, target_to_int_mapping, unique_targets = create_target_mapping(tasks[0]["histologies"])
    print(target_to_int_mapping)
    # exit()

    # target_mapping = get_oncotree_smoothed_target_mapping(target_mapping, target_to_int_mapping)
    if do_ontology_smoothing:
        print("Doing ontology smoothing")
        target_mapping = get_ncit_smoothed_target_mapping(
            "../data-commons/freeze/aeon_neighbors.json", target_to_int_mapping
        )

    assert len(target_mapping) == len(
        unique_targets
    ), f"Target mapping length {len(target_mapping)} != {len(unique_targets)}"

    assert len(target_mapping) == len(
        unique_targets
    ), f"Target mapping length {len(target_mapping)} != {len(unique_targets)}"
    assert len(target_mapping[unique_targets[0]]) == len(
        unique_targets
    ), f"Target mapping length {len(target_mapping[unique_targets[0]])} != {len(unique_targets)}"

    # sarcoma_specimens_to_move_to_underspecified_df = pd.read_csv(
    #     "/gpfs/mskmind_ess/boehmk/data-commons/NOS_assignments_20190122.txt", sep="\t"
    # )
    print(sample_df.split.value_counts())
    # sample_df.loc[
    #     sample_df["sample_id"].isin(sarcoma_specimens_to_move_to_underspecified_df["SAMPLE_ID"]), "split"
    # ] = "underspecified"
    print(sample_df.split.value_counts())
    dataframes = dict([(x, sample_df[sample_df["split"] == x]) for x in sample_df["split"].unique()])
    for split, df in dataframes.items():
        print(f"split: {split} {df['oncotree_code_aeon'].value_counts()}")
        if split != "underspecified":
            assert "CUPNOS" not in df["oncotree_code_aeon"].unique(), "CUPNOS in non-underspecified split"
            assert "CUP" not in df["oncotree_code_aeon"].unique(), "CUP in non-underspecified split"
        if split == "underspecified":
            assert "SFT" not in df["oncotree_code_aeon"].unique(), "SFT in underspecified split"
            assert "BLCA" not in df["oncotree_code_aeon"].unique(), "BLCA in underspecified split"
    assert_no_dataleak(dataframes)
    assert_no_duplicates(dataframes)

    print(target_to_int_mapping)

    print(
        f"Len of train: {len(dataframes['train'])}, len of val: {len(dataframes['val'])}, len of test: {len(dataframes['test'])}, len of underspecified: {len(dataframes['underspecified'])}, len of tcga: {len(dataframes['tcga'])}"
    )

    histologic_emb_dict = load_if_exists(histologic_emb, sample_df["oncotree_code_aeon"].unique().tolist())
    for oncotree_code in sample_df["oncotree_code"].unique():
        if "-" in oncotree_code:
            histologic_emb_dict[oncotree_code] = histologic_emb_dict[oncotree_code.split("-")[0]]
    if histologic_emb_dict:
        validate_embeddings(sample_df, histologic_emb_dict, colname="oncotree_code_aeon")

    for x, y in dataframes.items():
        assert "SEX" in y.columns, "SEX not in columns"
        assert "TISSUE_SITE" in y.columns, "TISSUE_SITE not in columns"

    print(f"histologic_emb_dict: {histologic_emb_dict}")
    datasets = dict(
        [
            (
                x,
                create_joint_dataset(
                    dataframes[x],
                    tile_tensor_url_col,
                    histologic_emb_dict,
                    None,
                    tasks,
                    max_seq_len,
                    target_mapping=target_mapping,
                    target_to_int_mapping=target_to_int_mapping,
                ),
            )
            for x in dataframes.keys()
        ]
    )
    print(f"made datasets: {list(datasets.keys())}")
    return datasets


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    desired_oncotree_codes = set(
        [
            "ACC",
            "ACRM",
            "ACYC",
            "ALUCA",
            "ANGS",
            "ANSC",
            "ARMM",
            "ARMS",
            "ASTR",
            "ATM",
            "BA",
            "BCC",
            "BLAD",
            "BLCA",
            "BMGCT",
            "CCOV",
            "CCRCC",
            "CESC",
            "CHDM",
            "CHRCC",
            "CHS",
            "COAD",
            "CSCC",
            "DA",
            "DASTR",
            "DDLS",
            "DES",
            "DSRCT",
            "ECAD",
            "EGC",
            "EHAE",
            "EHCH",
            "EMPD",
            "EOV",
            "EPIS",
            "EPM",
            "ERMS",
            "ES",
            "ESCA",
            "ESCC",
            "GBAD",
            "GBM",
            "GCCAP",
            "GEJ",
            "GIST",
            "GRCT",
            "HCC",
            "HGNEC",
            "HGSOC",
            "HNMUCM",
            "HNSC",
            "IDC",
            "IHCH",
            "ILC",
            "LGSOC",
            "LMS",
            "LUAD",
            "LUCA",
            "LUNE",
            "LUPC",
            "LUSC",
            "MAAP",
            "MACR",
            "MBC",
            "MCC",
            "MFH",
            "MFS",
            "MNG",
            "MOV",
            "MPNST",
            "MRLS",
            "NBL",
            "NPC",
            "NSGCT",
            "OCS",
            "ODG",
            "OPHSC",
            "OS",
            "PAAC",
            "PAAD",
            "PAMPCA",
            "PANET",
            "PAST",
            "PECOMA",
            "PEMESO",
            "PHC",
            "PLMESO",
            "PRAD",
            "PRCC",
            "PTAD",
            "RBL",
            "READ",
            "SBOV",
            "SBWDNET",
            "SCBC",
            "SCHW",
            "SCLC",
            "SDCA",
            "SEM",
            "SFT",
            "SKCM",
            "SSRCC",
            "STAD",
            "SYNS",
            "TAC",
            "THAP",
            "THHC",
            "THME",
            "THPA",
            "THPD",
            "THYC",
            "THYM",
            "UCCC",
            "UCP",
            "UCS",
            "UEC",
            "ULMS",
            "UM",
            "USC",
            "UTUC",
            "VMM",
            "VSC",
            "WDLS",
            "WT",
        ]
    )
    m: Dict[str, Dataset] = hydra.utils.instantiate(cfg.nn.data.dataset, _recursive_=False)
    print(type(m))
    print(list(m.keys()))
    # exit()

    for k, v in m.items():
        codes = []
        print(k, len(v))
        for item in v:
            codes.append(v.int_to_name_class_mapping[item["target_as_int"].item()])
        if k == "underspecified":
            print(v.int_to_name_class_mapping)
            print(v.df.oncotree_code_aeon.unique())
            print("SFT" in v.df.oncotree_code_aeon.unique())
            # exit()
            codes_wanted_in_underspecified_but_not_observed = set(histologies_to_move_to_underspecified) - set(codes)
            codes_observed_in_underspecified_but_not_wanted = set(codes) - set(histologies_to_move_to_underspecified)
            assert (
                codes_wanted_in_underspecified_but_not_observed == set()
            ), f"Codes wanted in underspecified but not observed: {codes_wanted_in_underspecified_but_not_observed}"
            assert (
                codes_observed_in_underspecified_but_not_wanted == set()
            ), f"Codes observed in underspecified but not wanted: {codes_observed_in_underspecified_but_not_wanted}"
        elif k == "tcga":
            extra_codes = set(codes) - desired_oncotree_codes
            assert extra_codes == set(), f"Extra codes in tcga: {extra_codes}"
        else:
            extra_codes = set(codes) - desired_oncotree_codes
            assert extra_codes == set(), f"Extra codes in {k}: {extra_codes}"
            missing_codes = desired_oncotree_codes - set(codes)
            assert missing_codes == set(), f"Missing codes in {k}: {missing_codes}"


if __name__ == "__main__":
    main()
