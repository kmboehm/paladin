from typing import Dict, List, Optional, Union

import hydra
import omegaconf
import torch
from torch import nn

from nn_core.common import PROJECT_ROOT

from paladin.modules.backbone import AttnMILBackbone, PassThroughBackbone, TransformerBackbone
from paladin.modules.config import (
    AttnMILAggregatorConfig,
    PassThroughAggregatorConfig,
    TransformerAggregatorConfig,
)


class AcontextualAggregator(nn.Module):
    def __init__(self, tile_emb_dim: int, num_targets: int, encoder_cfg: Optional[Dict] = None):
        """Acontextual Aggregator module without task information or positional embeddings.

        Args:
            tile_emb_dim (int): Tile embedding dimension.
            num_targets (int): Number of targets to predict.
            encoder_cfg (Dict): Encoder configuration as dictionary from hydra.
        """
        super(AcontextualAggregator, self).__init__()
        self.tile_emb_dim = tile_emb_dim
        self.latent_dim = encoder_cfg.encoder_embed_dim
        self.num_targets = num_targets

        encoder_cfg = hydra.utils.instantiate(encoder_cfg, _recursive_=False)
        self.aggregator, self.aggregator_type = self.instantiate_aggregator(encoder_cfg)
        self.tile_projector = nn.Linear(tile_emb_dim, self.latent_dim)
        self.head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, num_targets),
        )
        if self.aggregator_type in ["FlashTransformerBackbone", "TransformerBackbone"]:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.latent_dim))

    def instantiate_aggregator(self, encoder_cfg: Union[AttnMILAggregatorConfig]):
        """Instantiate aggregator.

        Args:
            encoder_cfg (Union[AttnMILAggregatorConfig]): Encoder configuration.

        Returns:
            nn.Module: Encoder module.
        """
        print(encoder_cfg)
        if isinstance(encoder_cfg, AttnMILAggregatorConfig):
            return AttnMILBackbone(encoder_cfg), "AttnMILBackbone"
        elif isinstance(encoder_cfg, TransformerAggregatorConfig):
            return TransformerBackbone(encoder_cfg), "FlashTransformerBackbone"
        elif isinstance(encoder_cfg, PassThroughAggregatorConfig):
            return PassThroughBackbone(), "PassThroughBackbone"
        else:
            raise ValueError(f"Unsupported encoder configuration: {encoder_cfg}")

    def forward(self, batch: Dict):
        """Forward.

        Args:
            batch (Dict): Batch containing tile_embeddings.

        Returns:
            Dict[str, torch.Tensor]: {"logits": [batch_size, num_targets], "whole_part_representation": [batch_size, latent_dim]}
        """
        tile_embeddings: torch.Tensor = batch["tile_tensor"]  # [batch_size, num_tiles, tile_emb_dim]
        tile_embeddings = self.tile_projector(tile_embeddings)

        if self.aggregator_type == "AttnMILBackbone":
            whole_part_representation = self._forward_attnmil(tile_embeddings)
        elif self.aggregator_type == "FlashTransformerBackbone":
            whole_part_representation = self._forward_flash_transformer(tile_embeddings)
        elif self.aggregator_type == "PassThroughBackbone":
            whole_part_representation = self._forward_pass_through(tile_embeddings)
        else:
            raise ValueError(f"Unsupported aggregator type: {self.aggregator_type}")

        logits = self.head(whole_part_representation)
        return {"logits": logits, "whole_part_representation": whole_part_representation}

    def _forward_attnmil(self, tile_embeddings: torch.Tensor):
        return self.aggregator(tile_embeddings)

    def _forward_pass_through(self, wsi_embedding: torch.Tensor):
        return self.aggregator(wsi_embedding)

    def _forward_flash_transformer(self, tile_embeddings: torch.Tensor):
        batch_size, num_tiles, _ = tile_embeddings.shape
        # Get aggregator dtype to ensure consistency (important for mixed precision)
        # aggregator_dtype = next(self.aggregator.parameters()).dtype
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # .to(dtype=tile_embeddings.dtype)
        tile_embeddings = tile_embeddings
        # print(f'converted cls_token dtype: {cls_token.dtype}')
        # print(f'converted tile_embeddings dtype: {tile_embeddings.dtype}')
        encoder_input = torch.cat([cls_token, tile_embeddings], dim=1)
        encoder_output = self.aggregator(encoder_input)
        whole_part_representation = encoder_output[:, 0]
        return whole_part_representation


class AeonLateAggregator(AcontextualAggregator):
    def __init__(self, encoder_cfg: Dict, tile_emb_dim: int, num_targets: int):
        """Aeon Aggregator module with primary vs metastasis provided along with slide, without positional embeddings.

        Args:
            encoder_cfg (Dict): Encoder configuration as dictionary from hydra.
            tile_emb_dim (int): Tile embedding dimension.
            num_targets (int): Number of targets to predict at a time. Must be 1.
        """
        super(AeonLateAggregator, self).__init__(tile_emb_dim, num_targets, encoder_cfg)
        mini_latent_dim = self.latent_dim // 4
        self.site_projector = nn.Linear(1, mini_latent_dim)
        self.sex_projector = nn.Embedding(3, mini_latent_dim)
        self.tissue_site_projector = nn.Embedding(57, mini_latent_dim)
        self.head = self.head = nn.Sequential(
            nn.Linear(self.latent_dim + mini_latent_dim * 3, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, num_targets),
        )

    def forward(self, batch: Dict):
        """Forward.

        Args:
            batch (Dict): Batch containing tile_embeddings.

        Returns:
            Dict[str, torch.Tensor]: {"logits": [batch_size, num_targets], "whole_part_representation": [batch_size, latent_dim]}
        """
        tile_embeddings: torch.Tensor = batch["tile_tensor"]  # [batch_size, num_tiles, tile_emb_dim]
        tile_embeddings = self.tile_projector(tile_embeddings)

        if self.aggregator_type == "FlashTransformerBackbone":
            whole_part_representation = self._forward_flash_transformer_add_token_one(tile_embeddings)
        else:
            raise ValueError(f"Unsupported aggregator type: {self.aggregator_type}")

        site = batch["site"]
        sex = batch["SEX"]
        tissue_site = batch["TISSUE_SITE"]
        batch_size, _, _ = tile_embeddings.shape
        assert all(
            s in ["Primary", "Metastasis"] for s in site
        ), f"AeonAggregator only supports 'Primary' or 'Metastasis' as site. Got {site}"
        site_integers = [int(s == "Metastasis") for s in site]
        site_integers = torch.tensor(site_integers, dtype=torch.float32, device=tile_embeddings.device).unsqueeze(
            1
        )  # [batch_size, 1, 1]
        site_integers = self.site_projector(site_integers)

        # Convert one-hot encodings to indices for embedding lookup
        sex_indices = torch.argmax(sex, dim=1)  # [batch_size]
        tissue_site_indices = torch.argmax(tissue_site, dim=1)  # [batch_size]

        sex_embedding = self.sex_projector(sex_indices)
        tissue_site_embedding = self.tissue_site_projector(tissue_site_indices)

        whole_part_representation = torch.cat(
            [whole_part_representation, site_integers, sex_embedding, tissue_site_embedding], dim=1
        )

        logits = self.head(whole_part_representation)
        return {"logits": logits, "whole_part_representation": whole_part_representation}

    def _forward_flash_transformer_add_token_one(self, tile_embeddings: torch.Tensor):
        batch_size, num_tiles, _ = tile_embeddings.shape
        cls_token = self.cls_token.expand(batch_size, -1, -1)

        encoder_input = torch.cat([cls_token, tile_embeddings], dim=1)
        encoder_output = self.aggregator(encoder_input)
        whole_part_representation = encoder_output[:, 0]
        return whole_part_representation

