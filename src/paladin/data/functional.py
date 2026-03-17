from typing import Dict, List

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset


def collate_fn(samples: List, split: str, metadata):
    """Custom collate function for dataloaders with access to split and metadata."""
    # Special keys that need tensor stacking
    tensor_keys = {"histologic_embedding", "target_embedding", "target", "SEX", "TISSUE_SITE", "treatment_integer"}

    # Initialize batch dict with empty lists for non-tensor keys
    batch = {key: [] for key, value in samples[0].items() if key not in tensor_keys and key != "tile_tensor"}

    # Collect all values in a single pass
    for item in samples:
        for key in batch:
            batch[key].append(item[key])

    # Handle tensor keys efficiently
    for key in tensor_keys:
        if key not in samples[0]:
            continue # for treatment_integer which is not always present
        if isinstance(samples[0][key], torch.Tensor):
            batch[key] = torch.stack([item[key] for item in samples], dim=0)
        else:
            batch[key] = [item[key] for item in samples]

    # Handle tile_tensor padding
    batch["tile_tensor"] = torch.nn.utils.rnn.pad_sequence([item["tile_tensor"] for item in samples], batch_first=True)

    return batch


def add_class_weights(train_dataset: Dataset) -> List[Dict]:
    """Add class imbalance information to target_dicts.

    Args:
        target_dicts: the target_dicts

    Returns:
        List[Dict]: the target_dicts with class imbalance information
    """
    task = train_dataset.task

    all_targets = train_dataset.report_targets()
    print(f"all_targets: {all_targets}")
    print(f"all_targets.shape: {all_targets.shape}")
    n_samples = len(train_dataset)
    print(f"n_samples: {n_samples}")
    pos_weights = []
    # Track column offset because survival targets occupy 2 columns (time, event)
    col_offset = 0
    for idx in range(len(task["target"])):
        if task["task"][idx] == "classification":
            print(f"calculating pos_weights for task {task['target'][idx]}")
            pos_weights.append(calc_class_weight(all_targets[:, col_offset], n_samples))
            col_offset += 1
        elif task["task"][idx] == "survival":
            print(f"skipping task {task['target'][idx]} because it is a survival task")
            col_offset += 2  # time + event columns
        else:
            print(f"skipping task {task['target'][idx]} because it is a {task['task'][idx]} task")
            col_offset += 1
    print(f"pos_weights: {pos_weights}")
    task["pos_weights"] = pos_weights
    return task


def calc_class_weight(target_values: torch.Tensor, n_samples: int) -> float:
    n_unique = len(torch.unique(target_values))
    assert n_unique == 2, f"Expected 2 unique values in target tensor, got {n_unique}: {target_values}"
    n_0 = torch.sum(target_values == 0).item()
    n_1 = torch.sum(target_values == 1).item()
    return n_0 / n_1


def add_class_weights_multiclass(
    train_dataset: Dataset,
    int_to_name_class_mapping: Dict,
) -> List[Dict]:
    target_dict = train_dataset.task
    assert len(target_dict["task"]) == 1, "This function is only supported for single target"
    assert (
        target_dict["task"][0] == "multiclass-classification"
    ), "This function is only supported for multiclass classification"
    all_targets = train_dataset.report_targets().numpy().reshape(-1)  # this will be a N vector

    # identify which targets in the int_to_name_class_mapping are not present in all_targets
    missing_targets = [target for target in int_to_name_class_mapping.keys() if target not in all_targets]
    print(f"missing_targets: {missing_targets}")
    named_missing_targets = [int_to_name_class_mapping[target] for target in missing_targets]
    print(f"missing_targets (named): {named_missing_targets}")
    # add at least one of each missing target to all_targets
    all_targets = np.concatenate([all_targets, np.array(missing_targets)])

    # convert all_targets to a 1D tensor
    class_weights = compute_class_weight("balanced", classes=np.unique(all_targets), y=all_targets)

    # set weights of missing targets to 0
    class_weights[missing_targets] = 0
    print(class_weights[missing_targets])

    target_dict["class_weights"] = class_weights.tolist()
    return [target_dict]
