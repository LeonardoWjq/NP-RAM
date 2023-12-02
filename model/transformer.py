import math
from typing import Union

import torch
import torch.nn as nn
from torch.nn import (Flatten, Linear, TransformerEncoder,
                      TransformerEncoderLayer)

from model.base import BaseFeatureExtractor


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.1,
                 max_len: int = 100):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)  # [seq_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class TransformerExtractor(BaseFeatureExtractor):
    def __init__(self,
                 state_dim: int,
                 feature_dim: int = 256,
                 seq_len=8,
                 dropout=0.1,
                 d_model=128,
                 dim_ff=128,
                 num_heads=8,
                 layer_count=3):
        super(TransformerExtractor, self).__init__(state_dim=state_dim,
                                                   feature_dim=feature_dim)

        self.d_model = d_model

        self.embedding = Linear(in_features=state_dim,
                                out_features=d_model)  # project obs dimension to d_model dimension

        self.pos_encoder = PositionalEncoding(d_model=d_model,
                                              dropout=dropout)

        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=num_heads,
                                                dim_feedforward=dim_ff,
                                                dropout=dropout,
                                                batch_first=True)  # define one layer of encoder multi-head attention

        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=layer_count)  # chain multiple layers of encoder multi-head attention

        self.flatten = Flatten(start_dim=1,
                               end_dim=-1)

        self.linear = Linear(in_features=d_model*seq_len,
                             out_features=feature_dim)  # project sequence * d_model dimension to feature dimension

    def forward(self, state: torch.Tensor, rgbd: Union[torch.Tensor, None] = None) -> torch.Tensor:
        state = self.embedding(state)*math.sqrt(self.d_model)
        state = self.pos_encoder(state)
        feature = self.encoder(state)
        feature = self.flatten(state)
        return self.linear(feature)

    @classmethod
    def get_name(self):
        return 'Transformer'

if __name__ == '__main__':
    model = TransformerExtractor(state_dim=42)
    print(model)
    state = torch.randn(10, 8, 42)
    image = torch.randn(10, 8, 128, 128)
    out = model(state, image)
    print(out.shape)
    print(TransformerExtractor.get_name())
