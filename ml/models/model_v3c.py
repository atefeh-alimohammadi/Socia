"""
Socia Pattern Transformer V3-C

Controlled ablation of V3-A.

V3-A:
    category embedding
    + temporal embedding
    + learned event-order embedding

V3-C:
    category embedding
    + temporal embedding

The ONLY architectural change from V3-A is removal of
the learned event-order embedding.

Everything else is unchanged:
    - RelativeTimeAttentionBias
    - ManualSelfAttentionLayer
    - temporal projection
    - transformer layers
    - AttentionPooling
    - classifier
    - finite-value checks
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


class SociaPatternTransformerV3C(nn.Module):

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

        # ------------------------------------------------------------------
        # IMPORTANT:
        # V3-A has:
        #
        # self.order_embedding = nn.Embedding(...)
        #
        # V3-C intentionally has NO order embedding.
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

        # --------------------------------------------------------------
        # Embeddings
        # --------------------------------------------------------------

        category_emb = self.category_embedding(
            category_ids
        )

        if check_finite:

            _check_finite(
                category_emb,
                "category_embedding",
            )

        temporal_emb = self.temporal_projection(
            temporal_features
        )

        if check_finite:

            _check_finite(
                temporal_emb,
                "temporal_projection",
            )

        # --------------------------------------------------------------
        # V3-C core ablation
        #
        # V3-A:
        # encoder_input =
        #     category_emb
        #     + temporal_emb
        #     + order_emb
        #
        # V3-C:
        # encoder_input =
        #     category_emb
        #     + temporal_emb
        # --------------------------------------------------------------

        encoder_input = (
            category_emb
            + temporal_emb
        )

        if check_finite:

            _check_finite(
                encoder_input,
                "encoder_input",
            )

        # --------------------------------------------------------------
        # Relative-time attention bias
        # EXACTLY same as V3-A / V2
        # --------------------------------------------------------------

        attn_bias = self.relative_time_bias(
            raw_delta_hours
        )

        if check_finite:

            _check_finite(
                attn_bias,
                "relative_time_bias",
            )

        # --------------------------------------------------------------
        # Transformer
        # --------------------------------------------------------------

        x = encoder_input

        for layer_index, layer in enumerate(
            self.layers
        ):

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

        # --------------------------------------------------------------
        # Attention pooling
        # --------------------------------------------------------------

        pooled, attention_weights = (
            self.attention_pooling(
                encoder_output
            )
        )

        if check_finite:

            _check_finite(
                pooled,
                "pooled_output",
            )

        # --------------------------------------------------------------
        # Classifier
        # --------------------------------------------------------------

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