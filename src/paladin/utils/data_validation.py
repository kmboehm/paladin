from typing import Dict, List

import pandas as pd


def validate_columns(df: pd.DataFrame, columns: List[str]) -> None:
    """Validate that the dataframe has the specified columns.

    Args:
        df (pd.DataFrame): The dataframe.
        columns (List[str]): The columns to validate.
    """
    for column in columns:
        assert column in df.columns, f"The dataframe must have a {column} column."


def assert_no_dataleak(dataframes: Dict[str, pd.DataFrame]) -> None:
    """Assert that there is no dataleak between the splits.

    Args:
        dataframes (Dict[str, pd.DataFrame]): The dataframes for train, val, and test.
    """
    train_image_ids_flattened = []
    for item in dataframes["train"]["image_id"].tolist():
        if isinstance(item, list):
            train_image_ids_flattened.extend(item)
        else:
            train_image_ids_flattened.append(item)
    val_image_ids_flattened = []
    for item in dataframes["val"]["image_id"].tolist():
        if isinstance(item, list):
            val_image_ids_flattened.extend(item)
        else:
            val_image_ids_flattened.append(item)
    test_image_ids_flattened = []
    for item in dataframes["test"]["image_id"].tolist():
        if isinstance(item, list):
            test_image_ids_flattened.extend(item)
        else:
            test_image_ids_flattened.append(item)
    assert not set(train_image_ids_flattened).intersection(
        set(val_image_ids_flattened)
    ), f"There must be no dataleak between the train and val splits but {set(train_image_ids_flattened).intersection(set(val_image_ids_flattened))} was found."
    assert not set(train_image_ids_flattened).intersection(
        set(test_image_ids_flattened)
    ), f"There must be no dataleak between the train and test splits but {set(train_image_ids_flattened).intersection(set(test_image_ids_flattened))} was found."
    assert not set(val_image_ids_flattened).intersection(
        set(test_image_ids_flattened)
    ), f"There must be no dataleak between the val and test splits but {set(val_image_ids_flattened).intersection(set(test_image_ids_flattened))} was found."

    assert not set(dataframes["train"]["sample_id"]).intersection(
        set(dataframes["val"]["sample_id"])
    ), f"There must be no dataleak between the train and val splits but {set(dataframes['train']['sample_id']).intersection(set(dataframes['val']['sample_id']))} was found."
    assert not set(dataframes["train"]["sample_id"]).intersection(
        set(dataframes["test"]["sample_id"])
    ), f"There must be no dataleak between the train and test splits but {set(dataframes['train']['sample_id']).intersection(set(dataframes['test']['sample_id']))} was found."
    assert not set(dataframes["val"]["sample_id"]).intersection(
        set(dataframes["test"]["sample_id"])
    ), f"There must be no dataleak between the val and test splits but {set(dataframes['val']['sample_id']).intersection(set(dataframes['test']['sample_id']))} was found."

    assert not set(dataframes["train"]["patient_id"]).intersection(
        set(dataframes["val"]["patient_id"])
    ), f"There must be no dataleak between the train and val splits but {set(dataframes['train']['patient_id']).intersection(set(dataframes['val']['patient_id']))} was found."
    assert not set(dataframes["train"]["patient_id"]).intersection(
        set(dataframes["test"]["patient_id"])
    ), f"There must be no dataleak between the train and test splits but {set(dataframes['train']['patient_id']).intersection(set(dataframes['test']['patient_id']))} was found."
    assert not set(dataframes["val"]["patient_id"]).intersection(
        set(dataframes["test"]["patient_id"])
    ), f"There must be no dataleak between the val and test splits but {set(dataframes['val']['patient_id']).intersection(set(dataframes['test']['patient_id']))} was found."


def assert_no_duplicates(dataframes: Dict[str, pd.DataFrame]) -> None:
    """Assert that there are no duplicates in the dataframes.

    Args:
        dataframes (Dict[str, pd.DataFrame]): The dataframes for train, val, and test.
    """
    # assert not dataframes["train"]["sample_id"].duplicated().any(), "There must be no duplicates in the train split."
    assert not dataframes["val"]["sample_id"].duplicated().any(), "There must be no duplicates in the val split."
    assert not dataframes["test"]["sample_id"].duplicated().any(), "There must be no duplicates in the test split."


def assert_positive_samples_exist(df: pd.DataFrame, target: str) -> None:
    """Assert that positive samples exist for the target.

    Args:
        df (pd.DataFrame): The dataframe.
        target (str): The target column.
    """
    for split in ["train", "val", "test"]:
        assert (
            df[df[target] > 0].shape[0] > 0
        ), f"There must be positive samples for the {target} target in the {split} split."
        assert not any(
            df[target].isna()
        ), f"There must be no NaN values for the {target} target in the {split} split, but {target} has NaN values."


def validate_target_type_and_range(df: pd.DataFrame, target: str) -> None:
    """Validate that the target column is of type float and in the range [0,1].

    Args:
        df (pd.DataFrame): The dataframe.
        target (str): The target name.
    """
    if "oncotree_code" in target.lower():
        return
    assert df[target].dtype == "float", f"The {target} column must be of type float but is of type {df[target].dtype}."
    assert (
        df[target].min() >= 0 and df[target].max() <= 1
    ), f"The {target} column must be in the range [0,1] but got {df[target].min()} and {df[target].max()}. {df[target]}"


def validate(df: pd.DataFrame, targets: List[Dict], tile_tensor_url_col: str) -> None:
    cols_to_enforce = [
        "image_id",
        "sample_id",
        tile_tensor_url_col,
        "patient_id",
        "oncotree_code",
        "filtered_tiles_h5_path",
    ]
    validate_columns(
        df,
        cols_to_enforce + [x for sublist in targets for x in sublist["target"]],
    )  # validate the dataframe columns exist

    assert len(df) > 0, "The dataframe must not be empty."

    for target_dict in targets:  # validate the target columns are of type float and in range [0,1]
        for target in target_dict["target"]:
            validate_target_type_and_range(df, target)
        # can add code here to scale ranges outside [0,1] for e.g. TMB score

    for target_dict in targets:  # ensure positive samples exist for each task in each split (for classification tasks)
        for target_task, target_name in zip(target_dict["task"], target_dict["target"]):
            if target_task == "classification":
                assert_positive_samples_exist(df, target_name)

    assert (
        not df["patient_id"].isna().any()
    ), f"There must be no NaN values for the patient_id column but {df.loc[df['patient_id'].isna(), ['patient_id', 'image_id']]}"


def validate_embeddings(df: pd.DataFrame, histologic_emb_dict: Dict, colname="oncotree_code") -> None:
    """Validate that the histologic embeddings are valid.

    Args:
        df (pd.DataFrame): The dataframe.
        histologic_emb_dict (Dict): The histologic embedding dictionary.
        colname (str): The column name of the histology code.
    """
    missing_keys = [x for x in df[colname].unique() if x not in histologic_emb_dict.keys()]
    if len(missing_keys) > 0:
        raise ValueError(
            f"The following histologies are missing from the histologic embedding dictionary: {missing_keys}"
        )
    print("validated embeddings")
