from typing import Dict, List

import hydra
import omegaconf
import pandas as pd
from torch.utils.data import Dataset

from nn_core.common import PROJECT_ROOT

from paladin.utils.data_prep import create_joint_dataset, split_delimiter_separated_data
from paladin.utils.data_validation import assert_no_dataleak, assert_no_duplicates


def setup_dataset(
    sample_df_path: str,
    tile_tensor_url_col: str,
    tile_emb_dim: int,
    max_seq_len: int,
    tasks: List[Dict],
    return_coordinates: bool = False,
    **kwargs,
) -> Dict[str, Dataset]:
    sample_df = pd.read_parquet(sample_df_path)
    sample_df["tile_tensor_url"] = sample_df[tile_tensor_url_col]
    sample_df = split_delimiter_separated_data(sample_df)
    assert "filtered_tiles_h5_path" in sample_df.columns, "filtered_tiles_h5_path column not found in sample_df"
    assert "split" in sample_df.columns, "split column not found in sample_df"

    if 'treatment_integer' in sample_df.columns:
        sample_df["treatment_integer"] = sample_df["treatment_integer"].astype(int)

    # Drop rows with NaN in any outcome column used by the configured tasks.
    # Survival targets are encoded as "time_col:event_col"; classification/regression
    # targets are plain column names.
    outcome_cols = []
    for task in tasks:
        for target in task["target"]:
            if ":" in target:
                time_col, event_col = target.split(":", 1)
                outcome_cols.extend([time_col, event_col])
            else:
                outcome_cols.append(target)
    outcome_cols = [c for c in outcome_cols if c in sample_df.columns]
    before = len(sample_df)
    sample_df = sample_df.dropna(subset=outcome_cols)
    print(f"Dropped {before - len(sample_df)} rows with NaN in outcome cols {outcome_cols} ({len(sample_df)} remaining)")

    # create the datasets
    print(sample_df["split"].value_counts())
    dataframes = {
        "val": sample_df[sample_df["split"] == "val"],
        "test": sample_df[sample_df["split"] == "test"],
        "train": sample_df[sample_df["split"] == "train"],
    }

    assert_no_dataleak(dataframes)
    assert_no_duplicates(dataframes)

    datasets = dict(
        [
            (
                x,
                create_joint_dataset(
                    dataframes[x],
                    tile_tensor_url_col,
                    None,
                    None,
                    tasks,
                    max_seq_len,
                    return_coordinates=False,
                    **kwargs,
                ),
            )
            for x in dataframes.keys()
        ]
    )
    print("made datasets")
    return datasets


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    m: Dict[str, Dataset] = hydra.utils.instantiate(cfg.nn.data.dataset, _recursive_=False)

    for k, v in m.items():
        print(k, len(v))
        for item in v:
            print(item)


if __name__ == "__main__":
    main()
