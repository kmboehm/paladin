import json
import os
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from paladin.data.dataset import PaladinDataset
from paladin.utils.data_validation import validate


def identify_targets_to_load(targets):
    target_cols_to_load = [t for sublist in targets for t in sublist if "+" not in t and ":" not in t]
    plus_targets = [t for sublist in targets for t in sublist if "+" in t]
    # Survival targets are formatted as "time_col:event_col"
    survival_targets = [t for sublist in targets for t in sublist if ":" in t]
    for target in plus_targets:
        target_cols_to_load.extend(target.split("+"))
    for target in survival_targets:
        time_col, event_col = target.split(":")
        target_cols_to_load.append(time_col)
        target_cols_to_load.append(event_col)
    target_cols_to_load = list(set(target_cols_to_load))

    if "oncotree_code" in target_cols_to_load:
        target_cols_to_load.remove("oncotree_code")
        target_cols_to_load.append("ONCOTREE_CODE")

    return target_cols_to_load, plus_targets


def generate_plus_targets(sample_df, plus_targets):
    for target in plus_targets:
        subtargets = target.split("+")
        print(f"Generating target for {target}")
        sample_df[target] = sample_df[subtargets[0]].astype(bool)
        for subtarget in subtargets[1:]:
            sample_df[target] = sample_df[target] & sample_df[subtarget].astype(bool)
        sample_df[target] = sample_df[target].astype(float)
        sample_df.loc[sample_df[subtargets].isna().any(axis=1), target] = np.nan
    return sample_df


def limit_to_sites(sample_df: pd.DataFrame, sites: List[List[str]]) -> pd.DataFrame:
    try:
        flattened_sites = [item for sublist in sites for item in sublist]
        print(f"Flattened sites: {flattened_sites}")
    except:  # noqa: E722
        print("Cannot flatten sites, assuming all allowed!")
        return sample_df

    unique_sites = []
    if sites is not None:
        for site_list in sites:
            for site in site_list:
                assert site in [
                    "Primary",
                    "Metastasis",
                    "Local Recurrence",
                    "Unknown",
                ], f"Site {site} is not supported; choose from Primary, Metastasis, Local Recurrence, Unknown."
                unique_sites.append(site)
    unique_sites = list(set(unique_sites))
    sample_df = sample_df[sample_df["site"].isin(unique_sites)]
    return sample_df


def limit_to_histologies(sample_df: pd.DataFrame, histologies: List[List[str]]) -> pd.DataFrame:
    try:
        flattened_histologies = [item for sublist in histologies for item in sublist]
    except:  # noqa: E722
        print("Cannot flatten histologies, assuming all allowed!")
        return sample_df

    if histologies is not None:
        for x in flattened_histologies:
            if x not in sample_df["oncotree_code"].unique():
                print(f"WARNING: Histology {x} not found.")
    sample_df = sample_df[sample_df["oncotree_code"].isin(flattened_histologies)]
    return sample_df


def split_delimiter_separated_data(sample_df):
    for col in sample_df.columns:
        if sample_df[col].astype(str).str.contains("|||", regex=False).any():
            sample_df[col] = sample_df[col].str.split("|||", regex=False)
    return sample_df


def create_joint_dataset(
    df: pd.DataFrame,
    tile_tensor_url_col: str,
    histologic_emb: Dict,
    target_emb: Dict,
    targets: List[Dict],
    max_seq_len: int,
    return_coordinates: bool = True,
    use_wds: bool = False,
    wds_shard_dir: Optional[str] = None,
    wds_shuffle_buffer: int = 150,  # Default matches DEFAULT_SHUFFLE_BUFFER_SIZE in wds_dataset
    wds_shuffle: bool = True,
    **kwargs,
) -> Dataset:
    """Create a dataset.

    Args:
        df (pd.DataFrame): The dataframe.
        tile_tensor_url_col (str): The column name with the URL to the tile tensor files.
        histologic_emb (Dict): The histologic embedding mapping.
        target_emb (Dict): The target embedding mapping.
        targets (List[Dict]): The targets.
        max_seq_len (int): The longest allowable sequence length.
        return_coordinates (bool): Whether to load and return tile coordinates.
        use_wds (bool): Whether to use WebDataset-based streaming.
        wds_shard_dir (str): Path to directory containing tar shards. Required when use_wds=True.
            Can be set via PALADIN_WDS_SHARD_DIR environment variable.
        wds_shuffle_buffer (int): Size of the shuffle buffer for WDS (default: 150).
        wds_shuffle (bool): Whether to shuffle (True for train, False for val/test).
        **kwargs: Additional arguments passed to dataset constructor.

    Returns:
        Dataset: PaladinWDSDataset if use_wds=True, PaladinDataset otherwise.

    Raises:
        ValueError: If use_wds=True and wds_shard_dir is None, empty, or not a valid directory.
        AssertionError: If df contains multiple splits.
    """
    assert (
        len(df.split.unique()) == 1
    ), f"The dataframe must have a single split, but has {len(df.split.unique())} splits: {df.split.unique()}"

    if use_wds:
        from pathlib import Path

        from paladin.data.wds_dataset import PaladinWDSDataset

        # Validate shard_dir before constructing PaladinWDSDataset
        if not wds_shard_dir or (isinstance(wds_shard_dir, str) and not wds_shard_dir.strip()):
            raise ValueError(
                "use_wds=True requires a valid wds_shard_dir. "
                "Set PALADIN_WDS_SHARD_DIR environment variable or provide wds_shard_dir in config. "
                f"Current value: {wds_shard_dir!r}"
            )

        shard_path = Path(wds_shard_dir)
        if not shard_path.exists():
            raise ValueError(
                f"use_wds=True requires wds_shard_dir to exist. "
                f"Directory not found: {shard_path} "
                f"(resolved from wds_shard_dir={wds_shard_dir!r})"
            )

        if not shard_path.is_dir():
            raise ValueError(
                f"use_wds=True requires wds_shard_dir to be a directory. "
                f"Path exists but is not a directory: {shard_path}"
            )

        return PaladinWDSDataset(
            df,
            targets,
            histologic_emb,
            target_emb,
            max_seq_len,
            shard_dir=wds_shard_dir,
            shuffle=wds_shuffle,
            shuffle_buffer=wds_shuffle_buffer,
            return_coordinates=return_coordinates,
            **kwargs,
        )

    return PaladinDataset(
        df, targets, histologic_emb, target_emb, max_seq_len, return_coordinates=return_coordinates, **kwargs
    )


PALADIN_COL_NAMES = [
    "image_id",
    "sample_id",
    "PATIENT_ID",
    "ONCOTREE_CODE",
    "split",
    "filtered_tiles_h5_path",
    "SAMPLE_TYPE",
]

RENAMED_PALADIN_COL_NAMES = [
    "image_id",
    "sample_id",
    "patient_id",
    "oncotree_code",
    "split",
    "filtered_tiles_h5_path",
    "tile_tensor_url",
    "site",
]


def rename_paladin_columns(sample_df, tile_tensor_url_col):
    sample_df = sample_df.rename(
        columns={
            tile_tensor_url_col: "tile_tensor_url",
            "ONCOTREE_CODE": "oncotree_code",
            "PATIENT_ID": "patient_id",
            "SAMPLE_TYPE": "site",
        }
    )
    return sample_df


def move_classification_first(tasks: List[Dict]) -> List[Dict]:
    """Move classification tasks to the front, then regression, then survival.

    Args:
        tasks (List[Dict]): The tasks as list of dictionaries, with each dict containing lists with keys "target", "target_type", "task", "sites", and "histologies".

    Returns:
        List[Dict]: The tasks with classification tasks moved to the front.
    """
    ordered_tasks = []
    for single_task_dict in tasks:
        ordered_single_task_dict = {
            "histologies": single_task_dict["histologies"],
            "sites": single_task_dict["sites"],
            "target": [],
            "target_type": [],
            "task": [],
        }
        reg_targets = []
        reg_target_types = []
        reg_tasks = []
        surv_targets = []
        surv_target_types = []
        surv_tasks = []
        for _target, _target_type, _task in zip(
            single_task_dict["target"], single_task_dict["target_type"], single_task_dict["task"]
        ):
            if _task == "regression":
                reg_targets.append(_target)
                reg_target_types.append(_target_type)
                reg_tasks.append(_task)
            elif _task == "survival":
                surv_targets.append(_target)
                surv_target_types.append(_target_type)
                surv_tasks.append(_task)
            else:
                ordered_single_task_dict["target"].append(_target)
                ordered_single_task_dict["target_type"].append(_target_type)
                ordered_single_task_dict["task"].append(_task)
        for _target, _target_type, _task in zip(reg_targets, reg_target_types, reg_tasks):
            ordered_single_task_dict["target"].append(_target)
            ordered_single_task_dict["target_type"].append(_target_type)
            ordered_single_task_dict["task"].append(_task)
        for _target, _target_type, _task in zip(surv_targets, surv_target_types, surv_tasks):
            ordered_single_task_dict["target"].append(_target)
            ordered_single_task_dict["target_type"].append(_target_type)
            ordered_single_task_dict["task"].append(_task)
        ordered_tasks.append(ordered_single_task_dict)
    return ordered_tasks


def cols_to_keep(target_cols_to_load: List[str], tile_tensor_url_col: Optional[str] = None, renamed=False) -> List[str]:
    if renamed:
        assert tile_tensor_url_col is None, "tile_tensor_url_col should be None when renamed is True"
        return RENAMED_PALADIN_COL_NAMES + target_cols_to_load
    else:
        return PALADIN_COL_NAMES + [tile_tensor_url_col] + target_cols_to_load


def prepare_run_specific_frame(
    sample_df_path, tile_tensor_url_col, tasks, drop_vus=False, limit_histologies=True, reef_dir=None
):
    # identify targets to load, including any targets with "+" that will be combined prior to dataset creation
    target_cols_to_load, plus_targets = identify_targets_to_load([x["target"] for x in tasks])
    target_cols_to_load.extend(["SEX", "METASTATIC_SITE_x", "PRIMARY_SITE_y"])
    original_target_cols_to_load = deepcopy(target_cols_to_load)
    if drop_vus:
        avail_cols_path = os.environ.get("PALADIN_AVAIL_COLS_PATH", "../data-commons/freeze/avail_cols.json")
        print(f"WARNING: using available columns at {avail_cols_path}")
        with open(avail_cols_path, "r") as f:
            avail_cols = json.load(f)
        final_target_cols_to_load = []
        oncogenic_targets_to_filter_vus_by = []

        for target_col in target_cols_to_load:
            if "_ONCOGENIC" in target_col:
                oncogenic_targets_to_filter_vus_by.append(target_col)
                final_target_cols_to_load.append(target_col.replace("_ONCOGENIC", ""))
                for suffix in ["_SV", "_HomDel", "_Amp", "_fusion"]:
                    variant_col = target_col.replace("_ONCOGENIC", suffix)
                    if variant_col in avail_cols:
                        final_target_cols_to_load.append(variant_col)
            final_target_cols_to_load.append(target_col)
        target_cols_to_load = final_target_cols_to_load
        print(target_cols_to_load)
    print(f"Reading {sample_df_path} with cols {target_cols_to_load}")

    sample_df = pd.read_parquet(
        sample_df_path,
        columns=cols_to_keep(target_cols_to_load, tile_tensor_url_col=tile_tensor_url_col),
        engine="fastparquet",
    )

    if drop_vus:
        vus_mask = np.zeros(len(sample_df), dtype=bool)
        for target_col in oncogenic_targets_to_filter_vus_by:
            print(f"dropping vus for {target_col}")
            base_col = target_col.replace("_ONCOGENIC", "")
            oncogenic_col = target_col
            # Start with indel/SNV VUS: oncogenic=0 but base mutation present
            single_target_vus_mask = (sample_df[oncogenic_col] == 0) & (sample_df[base_col] == 1)
            # Check each variant type for additional VUS
            for suffix in ["_SV", "_HomDel", "_Amp", "_fusion"]:
                variant_col = target_col.replace("_ONCOGENIC", suffix)
                if variant_col in sample_df.columns:
                    single_target_vus_mask |= (sample_df[oncogenic_col] == 0) & (sample_df[variant_col] == 1)
            vus_mask = vus_mask | single_target_vus_mask
        vus_mask = np.logical_and(vus_mask, sample_df["split"] != "tcga")
        sample_df.loc[vus_mask, "split"] = "vus"
        # print(f"Reassigned {vus_mask.sum()} VUS samples to 'vus' split")
        # print(f'number of fusions {fusion_mask.sum()}')
        # print(f'number of sv {sv_mask.sum()}')
        # print(f'number of amp {amp_mask.sum()}')
        # print(sample_df[amp_mask])
        # print(f'number of del {del_mask.sum()}')
        # print(f'number of indel_snv {indel_snv_mask.sum()}')
        # print(len(sample_df))
        # print(sample_df.groupby("split")[target_col].value_counts())

    sample_df = rename_paladin_columns(sample_df, tile_tensor_url_col)

    sample_df = generate_plus_targets(sample_df, plus_targets)

    sample_df = limit_to_sites(sample_df, [x["sites"] for x in tasks])
    if limit_histologies:
        sample_df = limit_to_histologies(sample_df, [x["histologies"] for x in tasks])

    dropna_cols = list(set(original_target_cols_to_load) - set(["SEX", "METASTATIC_SITE_x", "PRIMARY_SITE_y"]))
    sample_df = sample_df.dropna(subset=dropna_cols, how="any")  # drop samples with NaN for any task

    # split delimiter-separated data into lists of strings
    sample_df = split_delimiter_separated_data(sample_df)

    if reef_dir is not None:
        sample_df["tile_tensor_url"] = sample_df["tile_tensor_url"].apply(lambda x: [reef_dir + "/" + y for y in x])
        sample_df["filtered_tiles_h5_path"] = sample_df["filtered_tiles_h5_path"].apply(
            lambda x: [reef_dir + "/" + y for y in x]
        )

    # standardize and validate columns
    targets = move_classification_first(tasks)

    for target in target_cols_to_load:
        if (
            sample_df[target].dtype != "float"  # noqa: W503
            and "oncotree_code" not in target.lower()  # noqa: W503
            and "sex" not in target.lower()  # noqa: W503
            and "primary_site" not in target.lower()  # noqa: W503
            and "metastatic_site" not in target.lower()  # noqa: W503
        ):
            print(f"Converting {target} to float")
            sample_df[target] = sample_df[target].astype(float)

    validate(sample_df, targets, "tile_tensor_url")

    return sample_df, targets
