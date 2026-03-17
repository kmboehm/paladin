from typing import List

import torch
from torch.utils.data import Dataset


class InferenceDataset(Dataset):
    def __init__(
        self,
        sample_ids: List[str],
        sites: List[str],
        tile_tensor_urls: List[str],
        n_max_tiles: int = 20000,
    ) -> None:
        """Initialize the dataset.

        Args:
            sample_ids (List[str]): The sample ids.
            sites (List[str]): The site types ('Primary' or 'Metastasis').
            tile_tensor_urls (List[str]): The tile tensor urls as strings, with more than one url per item separated by "|||"

        Returns:
            None
        """
        self.sample_ids = sample_ids
        self.sites = sites
        self.tile_tensor_urls = tile_tensor_urls
        self.n_max_tiles = n_max_tiles
        assert len(sample_ids) == len(tile_tensor_urls), "sample_ids and tile_tensor_urls must have the same length"
        assert len(sample_ids) == len(sites), "sample_ids and sites must have the same length"
        assert all([val in ('Primary', 'Metastasis') for val in sites]), "the site value must be Primary or Metastasis"

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: the length of the dataset
        """
        return len(self.sample_ids)

    def assert_all_exist(self):
        for idx in range(len(self)):
            self.get_tile_tensor(self.tile_tensor_urls[idx])

    def get_tile_tensor(self, tile_tensor_url) -> torch.Tensor:
        """Get the tile tensor.

        Args:
            tile_tensor_url: the url of the tile tensor as str or list of str
            filtered_tiles_h5_url: the url of the filtered tiles h5 file as str
            image_ids: the image ids as list of str

        Returns:
            torch.Tensor: the tile tensor
        """
        url_or_urls = tile_tensor_url.split("|||")
        tile_tensor = []
        for tile_url in url_or_urls:
            emb = torch.load(tile_url, weights_only=True)
            # if emb.shape[0] > self.n_max_tiles:
            #     indices = torch.randperm(emb.shape[0])[: self.n_max_tiles]
            #     emb = emb[indices]
            tile_tensor.append(emb)
        tile_tensor = torch.cat(tile_tensor, dim=0)
        if tile_tensor.shape[0] > self.n_max_tiles:
            indices = torch.randperm(tile_tensor.shape[0])[: self.n_max_tiles]
            tile_tensor = tile_tensor[indices]
        if tile_tensor.shape[0] < self.n_max_tiles:
            padding = torch.zeros(self.n_max_tiles - tile_tensor.shape[0], tile_tensor.shape[1])
            tile_tensor = torch.cat([tile_tensor, padding], dim=0)
        return tile_tensor

    def __getitem__(self, idx: int) -> dict:
        """Return an item from the dataset.

        Args:
            idx: the index of the item to return

        Returns:
            dict: the item
        """
        sample_id = self.sample_ids[idx]
        site = self.sites[idx]
        tile_tensor_url = self.tile_tensor_urls[idx]
        tile_tensor = self.get_tile_tensor(tile_tensor_url)
        return {
            "sample_id": sample_id,
            "site": site,
            "tile_tensor": tile_tensor,
        }
