"""
Socia Pattern Transformer V3-B

V3-B = V3-A + a learned scalar temporal gate.

Everything else is intentionally kept identical to V3-A:
- same category embedding
- same temporal projection
- same learned order embedding
- same RelativeTimeAttentionBias
- same ManualSelfAttentionLayer
- same AttentionPooling
- same classifier
- same input/output interface

NEW in V3-B:
    A learned scalar gate per event controls how strongly the
    temporal embedding contributes to the encoder input.

    temporal_gate = sigmoid(Linear(temporal_features))

    encoder_input =
        category_emb
        + order_emb
        + temporal_gate * temporal_emb

The gate is initialized close to 1 so that the model starts
close to the original V3-A behavior rather than immediately
suppressing temporal information.

This file does not modify V3-A.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import (
    D_MODEL,
    DIM_FEEDFORWARD,
    DROPOUT,
    NHEAD,
    NUM_CATEGORIES,
    NUM_LAYERS,
    WINDOW_LENGTH,
    AttentionPooling,
)

from model_v2 import (
    RelativeTimeAttentionBias,
    ManualSelfAttentionLayer,
    TEMPORAL_FEATURE_DIM,
    _check_finite,
)


class SociaPatternTransformerV3B(nn.Module):

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

        # ------------------------------------------------------------------
        # Same as V3-A
        # ------------------------------------------------------------------

        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=d_model,
        )

        self.temporal_projection = nn.Linear(
            temporal_feature_dim,
            d_model,
        )

        self.order_embedding = nn.Embedding(
            num_embeddings=window_length,
            embedding_dim=d_model,
        )

        # ------------------------------------------------------------------
        # NEW in V3-B
        #
        # One scalar gate per event.
        #
        # Input:
        #   (B, L, temporal_feature_dim)
        #
        # Output:
        #   (B, L, 1)
        #
        # The scalar is broadcast over the D dimensions of temporal_emb.
        # ------------------------------------------------------------------

        self.temporal_gate = nn.Linear(
            temporal_feature_dim,
            1,
        )

        # Initialize the gate close to 1:
        #
        # sigmoid(2.0) ~= 0.881
        #
        # This keeps the starting point reasonably close to V3-A while
        # allowing the gate to learn during training.
        nn.init.zeros_(self.temporal_gate.weight)
        nn.init.constant_(self.temporal_gate.bias, 2.0)

        # ------------------------------------------------------------------
        # Same as V3-A
        # ------------------------------------------------------------------

        self.relative_time_bias = RelativeTimeAttentionBias(
            nhead=nhead
        )

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

        self.attention_pooling = AttentionPooling(
            d_model=d_model
        )

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
        temporal_features : (B, L, 4) float32, normalized
        raw_delta_hours   : (B, L) float32, unnormalized hours

        Returns:
            logits : (B,)

        If return_attention=True:
            logits, attention_weights, pooled
        """

        if check_finite:
            _check_finite(
                temporal_features,
                "temporal_features_input",
            )
            _check_finite(
                raw_delta_hours,
                "raw_delta_hours_input",
            )

        batch_size, seq_len = category_ids.shape

        # ------------------------------------------------------------------
        # Category embedding -- unchanged from V3-A
        # ------------------------------------------------------------------

        category_emb = self.category_embedding(
            category_ids
        )

        if check_finite:
            _check_finite(
                category_emb,
                "category_embedding",
            )

        # ------------------------------------------------------------------
        # Temporal projection -- unchanged from V3-A
        # ------------------------------------------------------------------

        temporal_emb = self.temporal_projection(
            temporal_features
        )

        if check_finite:
            _check_finite(
                temporal_emb,
                "temporal_projection",
            )

        # ------------------------------------------------------------------
        # Learned event-order embedding -- unchanged from V3-A
        # ------------------------------------------------------------------

        position_ids = torch.arange(
            seq_len,
            device=category_ids.device,
        ).unsqueeze(0).expand(batch_size, -1)

        order_emb = self.order_embedding(
            position_ids
        )

        if check_finite:
            _check_finite(
                order_emb,
                "order_embedding",
            )

        # ------------------------------------------------------------------
        # NEW: scalar temporal gate
        #
        # Shape:
        #   temporal_gate_values: (B, L, 1)
        #
        # Broadcasting:
        #   (B, L, 1) * (B, L, D)
        #       -> (B, L, D)
        # ------------------------------------------------------------------

        temporal_gate_values = torch.sigmoid(
            self.temporal_gate(temporal_features)
        )

        if check_finite:
            _check_finite(
                temporal_gate_values,
                "temporal_gate_values",
            )

        gated_temporal_emb = (
            temporal_gate_values * temporal_emb
        )

        if check_finite:
            _check_finite(
                gated_temporal_emb,
                "gated_temporal_embedding",
            )

        # ------------------------------------------------------------------
        # Main change relative to V3-A:
        #
        # V3-A:
        #   category_emb + temporal_emb + order_emb
        #
        # V3-B:
        #   category_emb + gated_temporal_emb + order_emb
        # ------------------------------------------------------------------

        encoder_input = (
            category_emb
            + gated_temporal_emb
            + order_emb
        )

        if check_finite:
            _check_finite(
                encoder_input,
                "encoder_input",
            )

        # ------------------------------------------------------------------
        # Relative-time attention bias -- unchanged from V3-A
        # ------------------------------------------------------------------

        attn_bias = self.relative_time_bias(
            raw_delta_hours
        )

        if check_finite:
            _check_finite(
                attn_bias,
                "relative_time_bias",
            )

        # ------------------------------------------------------------------
        # Transformer layers -- unchanged
        # ------------------------------------------------------------------

        x = encoder_input

        for layer_index, layer in enumerate(self.layers):

            x = layer(
                x,
                attn_bias,
                check_finite=check_finite,
            )

            if check_finite:
                _check_finite(
                    x,
                    f"encoder_layer_{layer_index}_output",
                )

        encoder_output = x

        # ------------------------------------------------------------------
        # Attention pooling -- unchanged
        # ------------------------------------------------------------------

        pooled, attention_weights = self.attention_pooling(
            encoder_output
        )

        if check_finite:
            _check_finite(
                pooled,
                "pooled_output",
            )

        # ------------------------------------------------------------------
        # Classifier -- unchanged
        # ------------------------------------------------------------------

        logits = self.classifier(
            pooled
        ).squeeze(-1)

        if check_finite:
            _check_finite(
                logits,
                "logits",
            )

        if return_attention:
            return (
                logits,
                attention_weights,
                pooled,
            )

        return logits