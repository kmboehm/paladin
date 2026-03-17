
class AttnMILAggregatorConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        assert hasattr(self, "num_hidden_layers")
        assert hasattr(self, "hidden_dim")
        assert hasattr(self, "encoder_embed_dim")
        assert self.num_hidden_layers >= 0
        assert self.hidden_dim > 0
        assert self.encoder_embed_dim > 0


class TransformerAggregatorConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        assert hasattr(self, "encoder_embed_dim")
        assert hasattr(self, "num_layers")
        assert hasattr(self, "num_heads")
        assert hasattr(self, "feedforward_dim")
        assert hasattr(self, "dropout")
        assert hasattr(self, "store_attn")
        assert self.encoder_embed_dim > 0
        assert self.num_layers > 0
        assert self.num_heads > 0
        assert self.feedforward_dim > 0
        assert 0 <= self.dropout < 1
        assert isinstance(self.store_attn, bool)

    def __repr__(self):
        return str(self.__dict__)


class PassThroughAggregatorConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)
