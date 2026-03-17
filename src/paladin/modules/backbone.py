import torch
import torch.nn as nn

from paladin.modules.config import AttnMILAggregatorConfig, TransformerAggregatorConfig
from paladin.modules.transformer import TransformerEncoderLayerWithAttention


class AttnMILBackbone(nn.Module):
    def __init__(self, aggregator_cfg: AttnMILAggregatorConfig):
        """Attention-based MIL aggregator module.

        Args:
            aggregator_cfg (AttnMILAggregatorConfig): Attention-based MIL aggregator configuration.
        """
        super(AttnMILBackbone, self).__init__()
        self.attn_module = self.make_attn_module(aggregator_cfg)

    @staticmethod
    def make_attn_module(aggregator_cfg: AttnMILAggregatorConfig):
        """Make attention module.

        Args:
            aggregator_cfg (AttnMILAggregatorConfig): Attention-based MIL aggregator configuration.

        Returns:
            nn.Module: Attention module.
        """
        # input layer
        layers = [nn.Linear(aggregator_cfg.encoder_embed_dim, aggregator_cfg.hidden_dim), nn.ReLU()]
        # hidden layers
        for _ in range(aggregator_cfg.num_hidden_layers):
            layers.extend([nn.Linear(aggregator_cfg.hidden_dim, aggregator_cfg.hidden_dim), nn.ReLU()])
        # output layer
        layers.extend([nn.Linear(aggregator_cfg.hidden_dim, 1), nn.Softmax(dim=1)])

        return nn.Sequential(*layers)

    def forward(self, tile_embeddings: torch.Tensor):
        """Forward.

        Args:
            tile_embeddings: [batch_size, num_tiles, input_dim]

        Returns:
            torch.Tensor: [batch_size, num_tiles]
        """
        attn_weights = self.attn_module(tile_embeddings)
        weighted_embeddings = attn_weights * tile_embeddings
        return weighted_embeddings.sum(dim=1)


class TransformerBackbone(nn.Module):
    def __init__(self, aggregator_cfg: TransformerAggregatorConfig):
        """Transformer-based aggregator module.

        Args:
            aggregator_cfg (TransformerAggregatorConfig): Transformer aggregator configuration.
        """
        super(TransformerBackbone, self).__init__()
        self.encoder_embed_dim = aggregator_cfg.encoder_embed_dim
        self.num_layers = aggregator_cfg.num_layers
        self.num_heads = aggregator_cfg.num_heads
        self.feedforward_dim = aggregator_cfg.feedforward_dim
        self.dropout = aggregator_cfg.dropout
        self.store_attn = aggregator_cfg.store_attn

        encoder_layer = TransformerEncoderLayerWithAttention(
            d_model=self.encoder_embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.feedforward_dim,
            dropout=self.dropout,
            batch_first=True,
            store_attn=self.store_attn,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

    def forward(self, tile_embeddings: torch.Tensor):
        """Forward.

        Args:
            tile_embeddings: [batch_size, num_tiles, encoder_embed_dim]

        Returns:
            torch.Tensor: [batch_size, encoder_embed_dim]
        """
        # Apply transformer encoder
        encoded_embeddings = self.transformer_encoder(tile_embeddings)

        return encoded_embeddings


class PassThroughBackbone(nn.Module):
    def __init__(self):
        """Pass-through aggregator module.

        Args:
            aggregator_cfg (PassThroughAggregatorConfig): Pass-through aggregator configuration.
        """
        super(PassThroughBackbone, self).__init__()

    def forward(self, wsi_embedding: torch.Tensor):
        """Forward.

        Args:
            wsi_embedding: [batch_size, 1, encoder_embed_dim]

        Returns:
            torch.Tensor: [batch_size, encoder_embed_dim]
        """
        return wsi_embedding.squeeze(1)


def main():
    backbone = TransformerBackbone(
        TransformerAggregatorConfig(encoder_embed_dim=128, num_layers=2, num_heads=8, feedforward_dim=256, dropout=0.1)
    )
    tile_embeddings = torch.randn(16, 10, 128)
    aggregated_embedding = backbone(tile_embeddings)
    print(aggregated_embedding.shape)


if __name__ == "__main__":
    main()
