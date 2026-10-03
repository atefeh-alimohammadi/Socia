"""
Socia Pattern Transformer V3-A + Bucketed Recurrence (Experiment 1)

Isolated ablation on top of V3-A: replaces the continuous
gap_same_category_norm + has_prev_same_category pair (indices 2, 3 of
V2/V3-A's 4-dim temporal_features) with a learned categorical
embedding over a fixed set of recurrence-gap buckets.

Design choice, and why: rather than feeding an integer bucket id
(0..6) into temporal_projection alongside abs_time_norm/gap_prev_norm,
recurrence_bucket_embedding is a SEPARATE nn.Embedding table, summed
into encoder_input the same way order_embedding is summed in in
model_v3a.py. This avoids imposing an artificial ordinal/linear
geometry over bucket index (see handoff report Section 37) and keeps
the change isolated to "how recurrence is represented" without
touching how abs_time_norm / gap_prev_norm are projected.

Unchanged from V3-A (model_v3a.SociaPatternTransformerV3A):
    - category_embedding
    - order_embedding (learned, keyed by position 0..24)
    - RelativeTimeAttentionBias (imported from model_v2, operates on
      raw_delta_hours only -- never touches temporal_features or
      recurrence buckets)
    - ManualSelfAttentionLayer x NUM_LAYERS (imported from model_v2)
    - AttentionPooling (imported from model)
    - classifier head

Changed from V3-A:
    - temporal_projection now maps a 2-dim vector
      [abs_time_norm, gap_prev_norm] instead of 4-dim (the two
      recurrence channels are removed from this pathway)
    - + a new recurrence_bucket_embedding, added into encoder_input
      as a fourth additive term

Recurrence bucket boundaries are FIXED / PREDEFINED (not fit from
train or test data), taken directly from the exploratory recurrence-
bucket analysis in the handoff report (Sections 26 and 36):

    NO_PRIOR, 0-6h, 6-24h, 24-72h, 72-168h, 168-336h, >336h

This is explicitly a predefined-boundary experiment, not a
data-driven-boundary one. No test-set information is used anywhere in
this file.

Does not modify model.py, model_v2.py, or model_v3a.py.
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
    _check_finite,
)

# Reduced temporal feature dim: only [abs_time_norm, gap_prev_norm].
REDUCED_TEMPORAL_FEATURE_DIM = 2

# --------------------------------------------------------------------
# Fixed / predefined recurrence buckets
# --------------------------------------------------------------------

NO_PRIOR_BUCKET = 0
BUCKET_0_6H = 1
BUCKET_6_24H = 2
BUCKET_24_72H = 3
BUCKET_72_168H = 4
BUCKET_168_336H = 5
BUCKET_336H_PLUS = 6

NUM_RECURRENCE_BUCKETS = 7

RECURRENCE_BUCKET_LABELS = [
    "NO_PRIOR", "0-6h", "6-24h", "24-72h", "72-168h", "168-336h", ">336h",
]

# Upper bound (inclusive) in hours for buckets 1..5. Anything beyond
# the last edge falls into BUCKET_336H_PLUS. NO_PRIOR is handled
# separately via the has_prev_same_category flag, not via this list.
# These are FIXED exploratory boundaries (handoff report Section 26),
# not fit from train or test data.
RECURRENCE_BUCKET_EDGES_HOURS = [6.0, 24.0, 72.0, 168.0, 336.0]


def recurrence_bucket_id(gap_hours: float, has_prev: bool) -> int:
    """
    Single source of truth for the raw-hours -> bucket-id mapping.
    Used by dataset construction (dataset_v3a_bucketed.py) and by any
    ablation code that needs to recompute bucket ids (e.g. forcing
    NO_PRIOR for the recurrence-removal condition).
    """

    if not has_prev:
        return NO_PRIOR_BUCKET

    for i, edge in enumerate(RECURRENCE_BUCKET_EDGES_HOURS):
        if gap_hours <= edge:
            return i + 1  # buckets 1..5

    return BUCKET_336H_PLUS


class SociaPatternTransformerV3ABucketedRecurrence(nn.Module):

    def __init__(
        self,
        num_categories: int = NUM_CATEGORIES,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        window_length: int = WINDOW_LENGTH,
        reduced_temporal_feature_dim: int = REDUCED_TEMPORAL_FEATURE_DIM,
        num_recurrence_buckets: int = NUM_RECURRENCE_BUCKETS,
    ) -> None:

        super().__init__()

        self.window_length = window_length

        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=d_model,
        )

        self.temporal_projection = nn.Linear(reduced_temporal_feature_dim, d_model)

        # NEW: learned embedding over fixed recurrence-gap buckets,
        # summed into encoder_input the same way order_embedding is.
        self.recurrence_bucket_embedding = nn.Embedding(
            num_embeddings=num_recurrence_buckets,
            embedding_dim=d_model,
        )

        # Unchanged from V3-A.
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
        recurrence_bucket_ids: torch.Tensor,
        return_attention: bool = False,
        check_finite: bool = False,
    ):
        """
        category_ids           : (B, L) long
        temporal_features       : (B, L, 2) float32
                                  [abs_time_norm, gap_prev_norm] --
                                  identical normalization/formula to
                                  V3-A, minus the two recurrence
                                  channels.
        raw_delta_hours         : (B, L) float32, unnormalized hours --
                                  unchanged use, relative-time bias only.
        recurrence_bucket_ids   : (B, L) long, values in
                                  [0, NUM_RECURRENCE_BUCKETS).
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

        recurrence_emb = self.recurrence_bucket_embedding(recurrence_bucket_ids)  # (B, L, D)

        if check_finite:
            _check_finite(recurrence_emb, "recurrence_bucket_embedding")

        position_ids = torch.arange(
            seq_len, device=category_ids.device
        ).unsqueeze(0).expand(batch_size, -1)  # (B, L)

        order_emb = self.order_embedding(position_ids)  # (B, L, D)

        if check_finite:
            _check_finite(order_emb, "order_embedding")

        encoder_input = category_emb + temporal_emb + recurrence_emb + order_emb  # (B, L, D)

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
