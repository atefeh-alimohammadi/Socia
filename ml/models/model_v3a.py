"""
Socia Pattern Transformer V3-A

Isolated ablation: V2 + a learned event-order embedding. Nothing else
changes. This file does not modify model_v2.py — it imports the
components that are unchanged (RelativeTimeAttentionBias,
ManualSelfAttentionLayer, the finite-value check) directly from it,
rather than duplicating them, so V2 and V3-A cannot silently drift
apart on the parts that are supposed to be identical.

What V3-A adds relative to V2:

    a learned nn.Embedding(WINDOW_LENGTH, D_MODEL) keyed by the
    event's position (0..24) within the window, summed into the
    encoder input alongside the category and temporal embeddings.

This is explicitly NOT V1's sinusoidal positional encoding (fixed,
non-learned) -- it is a fresh, trainable order signal, introduced to
test whether V2's complete removal of positional information was
itself limiting performance, independent of any auxiliary loss.

What V3-A does NOT add (deferred to later steps in this experiment
chain, per the agreed hierarchy):
    - no order-verification / ranking loss
    - no Time2Vec
    - no changes to RelativeTimeAttentionBias, ManualSelfAttentionLayer,
      AttentionPooling, or the classifier head
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .model import (
    D_MODEL,
    DIM_FEEDFORWARD,
    DROPOUT,
    NHEAD,
    NUM_CATEGORIES,
    NUM_LAYERS,
    WINDOW_LENGTH,
    AttentionPooling,
)

from .model_v2 import (
    RelativeTimeAttentionBias,
    ManualSelfAttentionLayer,
    TEMPORAL_FEATURE_DIM,
    _check_finite,
)


class SociaPatternTransformerV3A(nn.Module):

    def __init__(
        self,
        num_categories: int = NUM_CATEGORIES,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        window_length: int = WINDOW_LENGTH,
        temporal_feature_dim: int = TEMPORAL_FEATURE_DIM,
    ) -> None:

        super().__init__()

        self.window_length = window_length

        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=d_model,
        )

        self.temporal_projection = nn.Linear(temporal_feature_dim, d_model)

        # NEW in V3-A: learned, trainable order embedding keyed by
        # position within the window. Distinct from V1's fixed
        # sinusoidal positional encoding -- this is a free parameter
        # the model must learn to use, not a hand-designed signal.
        self.order_embedding = nn.Embedding(
            num_embeddings=window_length,
            embedding_dim=d_model,
        )

        self.relative_time_bias = RelativeTimeAttentionBias(nhead=nhead)

        self.layers = nn.ModuleList(
            [
                ManualSelfAttentionLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
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
        temporal_features: torch.Tensor,
        raw_delta_hours: torch.Tensor,
        return_attention: bool = False,
        check_finite: bool = False,
    ):
        """
        category_ids      : (B, L) long
        temporal_features  : (B, L, 4) float32, normalized (same as V2)
        raw_delta_hours    : (B, L) float32, unnormalized hours --
                             used only for the relative-time attention
                             bias, unchanged from V2.
        """

        if check_finite:
            _check_finite(temporal_features, "temporal_features_input")
            _check_finite(raw_delta_hours, "raw_delta_hours_input")

        batch_size, seq_len = category_ids.shape

        category_emb = self.category_embedding(category_ids)  # (B, L, D)

        if check_finite:
            _check_finite(category_emb, "category_embedding")

        temporal_emb = self.temporal_projection(temporal_features)  # (B, L, D)

        if check_finite:
            _check_finite(temporal_emb, "temporal_projection")

        # NEW: learned order embedding, same position indices for
        # every window in the batch.
        position_ids = torch.arange(
            seq_len, device=category_ids.device
        ).unsqueeze(0).expand(batch_size, -1)  # (B, L)

        order_emb = self.order_embedding(position_ids)  # (B, L, D)

        if check_finite:
            _check_finite(order_emb, "order_embedding")

        encoder_input = category_emb + temporal_emb + order_emb  # (B, L, D)

        attn_bias = self.relative_time_bias(raw_delta_hours)  # (B, nhead, L, L)

        if check_finite:
            _check_finite(attn_bias, "relative_time_bias")

        x = encoder_input

        for layer_index, layer in enumerate(self.layers):
            x = layer(x, attn_bias, check_finite=check_finite)

            if check_finite:
                _check_finite(x, f"encoder_layer_{layer_index}_output")

        encoder_output = x

        pooled, attention_weights = self.attention_pooling(encoder_output)

        if check_finite:
            _check_finite(pooled, "pooled_output")

        logits = self.classifier(pooled).squeeze(-1)  # (B,)

        if check_finite:
            _check_finite(logits, "logits")

        if return_attention:
            return logits, attention_weights, pooled

        return logits