from typing import Optional

import torch.nn as nn
from torch import Tensor


class TransformerEncoderLayerWithAttention(nn.TransformerEncoderLayer):
    def __init__(self, *args, store_attn=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.store_attn = store_attn

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=not self.training and self.store_attn,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)
