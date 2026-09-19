from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


D_MODEL = 96
NHEAD = 6
NUM_LAYERS = 2
DIM_FEEDFORWARD = 192
DROPOUT = 0.2
WINDOW_LENGTH = 25
NUM_CATEGORIES = 15


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learned) sinusoidal positional encoding, shape (WINDOW_LENGTH, D_MODEL).
    Registered as a buffer so it moves with .to(device) but is not a trained parameter.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        max_len: int = WINDOW_LENGTH,
    ) -> None:

        super().__init__()

        position = torch.arange(max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, batch_size: int) -> torch.Tensor:

        return self.pe.unsqueeze(0).expand(batch_size, -1, -1)


class AttentionPooling(nn.Module):
    """
    Learned query vector attends over the encoder output sequence,
    producing a single pooled representation per window plus the
    attention weights (useful later for interpretability analysis
    against known true event positions).
    """

    def __init__(self, d_model: int = D_MODEL) -> None:

        super().__init__()

        self.query = nn.Parameter(torch.randn(d_model))

    def forward(
        self,
        encoder_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # encoder_output: (B, 25, D_MODEL)

        batch_size = encoder_output.size(0)

        query = self.query.unsqueeze(0).expand(batch_size, -1)
        # query: (B, D_MODEL)

        scores = torch.bmm(
            encoder_output,
            query.unsqueeze(2),
        ).squeeze(2)
        # scores: (B, 25)

        attention_weights = torch.softmax(scores, dim=1)
        # attention_weights: (B, 25)

        pooled = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_output,
        ).squeeze(1)
        # pooled: (B, D_MODEL)

        return pooled, attention_weights


class SociaPatternTransformer(nn.Module):

    def __init__(
        self,
        num_categories: int = NUM_CATEGORIES,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        window_length: int = WINDOW_LENGTH,
    ) -> None:

        super().__init__()

        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=d_model,
        )

        # delta_hours arrives already log1p'd and normalized by the
        # Dataset. This is a plain scalar-to-vector projection only -
        # no normalization or log transform happens here.
        self.time_projection = nn.Linear(1, d_model)

        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=window_length,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.attention_pooling = AttentionPooling(d_model=d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        category_ids: torch.Tensor,
        delta_hours: torch.Tensor,
        return_attention: bool = False,
    ):

        # category_ids: (B, 25) long
        # delta_hours:  (B, 25) float32, already log1p + normalized

        batch_size = category_ids.size(0)

        category_emb = self.category_embedding(category_ids)
        # category_emb: (B, 25, D_MODEL)

        time_emb = self.time_projection(delta_hours.unsqueeze(-1))
        # time_emb: (B, 25, D_MODEL)

        pos_emb = self.positional_encoding(batch_size)
        # pos_emb: (B, 25, D_MODEL)

        encoder_input = category_emb + time_emb + pos_emb
        # encoder_input: (B, 25, D_MODEL)

        encoder_output = self.encoder(encoder_input)
        # encoder_output: (B, 25, D_MODEL)

        pooled, attention_weights = self.attention_pooling(encoder_output)
        # pooled: (B, D_MODEL) | attention_weights: (B, 25)

        logits = self.classifier(pooled).squeeze(-1)
        # logits: (B,)

        if return_attention:
            return logits, attention_weights, pooled

        return logits