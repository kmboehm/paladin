from functools import cached_property
from typing import Dict, List, Optional

import h5py
import hydra
import omegaconf
import pandas as pd
import torch
from torch.utils.data import Dataset

from nn_core.common import PROJECT_ROOT


class PaladinDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tasks: List[Dict],
        histologic_emb: Optional[Dict],
        target_emb: Optional[Dict],
        max_seq_len: int,
        project_name: Optional[str] = "paladin",
        target_mapping: Optional[Dict] = None,
        return_coordinates: bool = False,
        target_to_int_mapping: Optional[Dict] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            df (pd.DataFrame): The dataframe containing the dataset.
            tasks (List[Dict]): A list of dictionaries containing task information.
            histologic_emb (Optional[Dict]): A dictionary containing histologic embeddings, if any.
            target_emb (Optional[Dict]): A dictionary containing target embeddings, if any.
            max_seq_len (int): The maximum sequence length.
            project_name (Optional[str]): The project name, if any - to override by the dataset config (not really used here)
            target_mapping (Optional[Dict]): A dictionary containing target mapping (str --> Iterable[float]), if any.
            target_to_int_mapping (Optional[Dict]): A dictionary containing target to int mapping (str --> int), if any.

        Returns:
            None
        """
        assert (
            len(tasks) == 1
        ), "Only one task per dataset is supported; for multiple tasks, please use setup_meta_datasets.py, Note that multi-task is allowed but meta tasks are not."
        self.task = tasks[0]
        self.df = df.copy().reset_index()
        self.histologic_emb_dict = histologic_emb
        self.target_emb_dict = target_emb
        self.max_seq_len = max_seq_len
        self.target_names = self.task["target"]

        self.target_mapping = target_mapping
        self.target_to_int_mapping = target_to_int_mapping
        self.return_coordinates = return_coordinates

        if self.target_mapping is not None:
            assert (
                self.target_to_int_mapping is not None
            ), "target_to_int_mapping must be provided if target_mapping is provided"
            reverse_target_mapping = dict()
            for key, value in self.target_to_int_mapping.items():
                reverse_target_mapping[value] = key
            self.int_to_name_class_mapping = reverse_target_mapping

    @cached_property
    def task_name(self) -> str:
        """Make the task name.

        Returns:
            str: the task name
        """
        if len(self.task["histologies"]) > 3:
            hist_str = "multi"
        else:
            hist_str = "-".join(self.task["histologies"])
        if len(self.task["sites"]) > 3:
            sites_str = "multi"
        else:
            sites_str = "-".join(self.task["sites"])
        if isinstance(self.task["target"], list):
            target_str = "-".join(self.task["target"])
        else:
            target_str = self.task["target"]
        return "_".join([hist_str, sites_str, target_str])

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: the length of the dataset
        """
        return len(self.df)

    def get_histologic_embedding(self, oncotree_code: str) -> Optional[torch.Tensor]:
        """Get the histologic embedding for the oncotree code.

        Args:
            oncotree_code: the oncotree code

        Returns:
            Optional[torch.Tensor]: the histologic embedding
        """
        if self.histologic_emb_dict is None:
            return None
        return self.histologic_emb_dict[oncotree_code]

    def get_target_embedding(self) -> Optional[torch.Tensor]:
        """Get the target embedding.

        Returns:
            Optional[torch.Tensor]: the target embedding
        """
        if self.target_emb_dict is None:
            return None
        emb = []
        for target_name in self.target_names:
            emb.append(self.target_emb_dict[target_name])
        return torch.stack(emb, dim=0)

    def get_tile_tensor(self, tile_tensor_url, filtered_tiles_h5_url, image_ids) -> torch.Tensor:
        """Get the tile tensor.

        Args:
            tile_tensor_url: the url of the tile tensor as str or list of str
            filtered_tiles_h5_url: the url of the filtered tiles h5 file as str
            image_ids: the image ids as list of str

        Returns:
            torch.Tensor: the tile tensor
        """
        # return None, None, None, None
        if isinstance(tile_tensor_url, str):
            url_or_urls = [tile_tensor_url]
            tile_tensor, coordinates_by_token = self.load_tiletensor_with_coordinates(
                tile_tensor_url, filtered_tiles_h5_url
            )
            image_ids_by_token = torch.tensor([int(image_ids[0])] * len(tile_tensor))
        elif isinstance(tile_tensor_url, list):
            url_or_urls = tile_tensor_url
            tile_tensor = []
            coordinates_by_token = []
            image_ids_by_token = []
            for tile_url, filt_feat_url, image_id in zip(tile_tensor_url, filtered_tiles_h5_url, image_ids):
                e, c = self.load_tiletensor_with_coordinates(tile_url, filt_feat_url)
                tile_tensor.append(e)
                image_ids_by_token.append(torch.tensor([int(image_id)] * len(e)))
                c = torch.tensor(c)
                if len(c.shape) == 1:
                    c = c.unsqueeze(0)
                coordinates_by_token.append(c)
            tile_tensor = torch.cat(tile_tensor, dim=0)
            image_ids_by_token = torch.cat(image_ids_by_token, dim=0)
            coordinates_by_token = torch.cat(coordinates_by_token, dim=0)
        else:
            raise ValueError(f"tile_tensor_url must be str or list of str, got {type(tile_tensor_url)}")
        if len(tile_tensor) > self.max_seq_len:
            idx = torch.randperm(len(tile_tensor))[: self.max_seq_len]
            tile_tensor = tile_tensor[idx]
            image_ids_by_token = image_ids_by_token[idx]
            coordinates_by_token = coordinates_by_token[idx]
        return url_or_urls, tile_tensor, image_ids_by_token, coordinates_by_token

    def get_tile_tensor_without_coordinates(self, tile_tensor_url) -> torch.Tensor:
        if isinstance(tile_tensor_url, str):
            tile_tensor = self.load_tiletensor_without_coordinates(tile_tensor_url)
        elif isinstance(tile_tensor_url, list):
            tile_tensor = []
            for tile_url in tile_tensor_url:
                e = self.load_tiletensor_without_coordinates(tile_url)
                tile_tensor.append(e)
            tile_tensor = torch.cat(tile_tensor, dim=0)
        else:
            raise ValueError(f"tile_tensor_url must be str or list of str, got {type(tile_tensor_url)}")
        if len(tile_tensor) > self.max_seq_len:
            idx = torch.randperm(len(tile_tensor))[: self.max_seq_len]
            tile_tensor = tile_tensor[idx]
        return tile_tensor

    def load_tiletensor_without_coordinates(self, tile_tensor_url: str) -> torch.Tensor:
        emb, _ = self._load_tile_embedding(tile_tensor_url)
        if len(emb.shape) == 1:
            emb = emb.unsqueeze(0)  # for WSI embedding case
        return emb

    def load_tiletensor_with_coordinates(self, tile_tensor_url: str, filtered_tiles_h5_url: str) -> torch.Tensor:
        """Load tile embeddings and corresponding coordinates, subsampled with the same indices."""
        emb, idx = self._load_tile_embedding(tile_tensor_url)
        coordinates_by_token = h5py.File(filtered_tiles_h5_url, "r")["coords"][:]
        coordinates_by_token = coordinates_by_token[idx]
        return emb, coordinates_by_token

    def _load_tile_embedding(self, tile_tensor_url: str):
        """Load and subsample a tile embedding tensor.

        Returns:
            tuple: (subsampled_embedding, indices_used)
        """
        try:
            try:
                emb = torch.load(tile_tensor_url, weights_only=True)
            except EOFError:
                raise RuntimeError(f"EOFError loading tensor from {tile_tensor_url}")
            idx = torch.randperm(len(emb))[: self.max_seq_len]
            emb = emb[idx]
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {tile_tensor_url}")
        except RuntimeError as e:
            raise RuntimeError(f"RuntimeError loading tensor from {tile_tensor_url}: {e}")
        return emb, idx

    def get_image_ids(self, image_id) -> List[str]:
        """Get the image ids.

        Args:
            image_id: the image id as str or list of str

        Returns:
            List[str]: the image ids
        """
        if isinstance(image_id, str):
            return [image_id]
        elif isinstance(image_id, list):
            return image_id
        else:
            raise ValueError(f"image_id must be str or list of str, got {type(image_id)}, specifically {image_id}")

    @cached_property
    def report_target_types(self) -> List[str]:
        """Report the target types.

        Returns:
            List[str]: the target types
        """
        return [self.task["target_type"]]

    def report_targets(self) -> torch.tensor:
        """Report the targets.

        Returns:
            torch.tensor: the targets
        """
        all_targets = []
        for idx in range(len(self)):
            all_targets.append(self.get_target_as_tensor(self.df.iloc[idx], target_as_int=True))
        return torch.stack(all_targets, dim=0)

    @cached_property
    def longest_sequence(self) -> int:
        """Report the longest sequence.

        Returns:
            int: the longest sequence of tiles
        """
        return self.max_seq_len

    def __getitem__(self, idx: int) -> dict:
        """Return an item from the dataset.

        Args:
            idx: the index of the item to return

        Returns:
            dict: the item
        """
        row = self.df.iloc[idx]
        patient_id = row["patient_id"]
        sample_id = row["sample_id"]
        try:
            oncotree_code = row["oncotree_code_aeon"]
        except KeyError:
            oncotree_code = row["oncotree_code"]
        histologic_embedding = self.get_histologic_embedding(oncotree_code)
        target_embedding = self.get_target_embedding()
        unique_image_ids = self.get_image_ids(row["image_id"])
        target = self.get_target_as_tensor(row)
        if self.return_coordinates:
            unique_tile_tensor_urls, tile_tensor, image_ids_by_token, coordinates_by_token = self.get_tile_tensor(
                row["tile_tensor_url"], row["filtered_tiles_h5_path"], unique_image_ids
            )
            tile_h5_urls = row["filtered_tiles_h5_path"]
        else:
            tile_tensor = self.get_tile_tensor_without_coordinates(row["tile_tensor_url"])
            unique_tile_tensor_urls = None
            image_ids_by_token = None
            coordinates_by_token = None
            tile_h5_urls = None
        site = row["site"]
        if self.target_to_int_mapping is not None:
            target_as_int = self.get_target_as_tensor(row, target_as_int=True)
        else:
            target_as_int = None
        return {
            "patient_id": patient_id,
            "sample_id": sample_id,
            "oncotree_code": oncotree_code,
            "histologic_embedding": histologic_embedding,
            "target_embedding": target_embedding,
            "image_ids": unique_image_ids,
            "image_ids_by_token": image_ids_by_token,  # batch_size, seq_length-1
            "coordinates_by_token": coordinates_by_token,  # batch_size, seq_length-1, 2
            "tile_tensor_urls": unique_tile_tensor_urls,
            "tile_tensor": tile_tensor,
            "target": target,
            "SEX": row["SEX"] if "SEX" in row else None,
            "TISSUE_SITE": row["TISSUE_SITE"] if "TISSUE_SITE" in row else None,
            "target_as_int": target_as_int,  # for aeon support
            "tiles_h5_urlpath": tile_h5_urls,
            "site": site,
            "target_names": self.target_names,
            "split": row["split"],
        }

    def get_target_as_tensor(self, row, target_as_int=False) -> torch.Tensor:
        """Get the target as tensor.

        For survival targets (formatted as 'time_col:event_col'), returns [time, event] per task.

        Returns:
            torch.Tensor: the targets as tensor
            row: the row in the dataframe
        """
        if self.target_mapping is None:
            targets = []
            for target_name in self.target_names:
                if ":" in target_name:
                    # Survival target: "time_col:event_col" → [time, event]
                    time_col, event_col = target_name.split(":")
                    targets.append(torch.Tensor([row[time_col], row[event_col]]).float())
                else:
                    targets.append(torch.Tensor([row[target_name]]).float().view(-1))
            return torch.cat(targets, dim=0)
        else:
            assert len(self.target_names) == 1, "Only one target name is supported for target mapping"
            if target_as_int:
                return torch.Tensor([self.target_to_int_mapping[row[self.target_names[0]]]]).int().view(-1)
            else:
                return (
                    self.target_mapping[row[self.target_names[0]]].float().view(-1)
                )  # this will already be a 1D tensor


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    _: Dataset = hydra.utils.instantiate(cfg.nn.data.dataset, _recursive_=False)
    for batch in _["train"]:
        print(batch)
        break


if __name__ == "__main__":
    main()
