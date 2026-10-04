"""
Socia Pattern Transformer V2

Changes relative to V1 (model.py), and why:

1. Per-event input now includes explicit relational temporal features
   (gap_prev, gap_same_category, has_prev_same_category) in addition
   to normalized absolute time -- V1 only had absolute time, which
   diagnostics showed the model was using as a coarse density/scale
   signal rather than as structured pattern information.

2. Sinusoidal positional encoding is REMOVED, not kept alongside the
   new components. V1's order-shuffle ablation showed the model was
   effectively order-invariant despite having positional encoding
   available, and keeping an index-based position signal would let
   V2 fall back on "which slot" instead of "how much real time
   elapsed" -- undermining the point of the change below.

3. A learned relative-time attention bias replaces positional
   encoding as the mechanism for representing order and spacing. For
   every pair of events (i, j) in a window, the SIGNED, real elapsed
   time between them is fed through a small MLP to produce a bias
   added directly to the attention logits for that pair, per head.

4. Pooling is unchanged (learned attention pooling, reused from V1).

---------------------------------------------------------------------
IMPLEMENTATION NOTE -- why this does NOT use nn.TransformerEncoder:

The first version of this file built the relative-time bias as a
(B*nhead, L, L) mask and passed it to nn.TransformerEncoder's `mask`
argument. That triggered NaN validation loss from epoch 1 onward,
while training loss stayed finite. Root cause:

nn.TransformerEncoderLayer contains an internal fused "fast path"
(torch._transformer_encoder_layer_fwd) that PyTorch switches to
automatically when grad tracking is off for all relevant tensors --
which is exactly the situation in a `torch.no_grad()` validation
loop. During training, gradients are enabled, so the fast path is
skipped and the normal eager-mode attention implementation runs
(which handles a custom (B*nhead, L, L) additive float mask
correctly). During validation, the fast path silently engages, and
its fused kernel does not reliably support an arbitrary per-head 3D
additive mask the way the eager path does, producing NaNs.

V1's model.py never hit this because it never passes a `mask` at all
(mask=None) -- the fast path has nothing incompatible to trip on
there. It's specific to V2's custom mask.

The fix here is to not use nn.TransformerEncoder /
nn.TransformerEncoderLayer at all. Instead, ManualSelfAttentionLayer
below reimplements the same architecture (post-norm, ReLU-activated
feedforward, same dropout placement as PyTorch's default
TransformerEncoderLayer) using only plain nn.Linear / nn.LayerNorm /
torch.matmul / torch.softmax -- no fused/fast-path kernels anywhere,
so behavior is guaranteed identical whether or not gradients are
being tracked, and there is no dependency on any PyTorch-version-
specific API (unlike e.g. torch.backends.mha.set_fastpath_enabled,
which only exists in some versions).
---------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

# Reuse V1's backbone hyperparameters and the (unchanged) attention
# pooling module directly, rather than redefining them.
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

TEMPORAL_FEATURE_DIM = 4  # [abs_norm, gap_prev_norm, gap_same_cat_norm, has_prev_flag]
REL_TIME_BIAS_HIDDEN_DIM = 16


# =====================================================================
# DEBUG UTILITY
# =====================================================================

def _check_finite(tensor: torch.Tensor, name: str) -> None:
    """
    Raise immediately with a clear, localized error if `tensor`
    contains any NaN/Inf. Used to pinpoint exactly which stage of the
    forward pass first produces non-finite values, without having to
    guess. Cheap (a single isfinite + all() call on small tensors),
    but still gated behind check_finite=False by default so it never
    adds overhead to the normal training loop.
    """

    if not torch.isfinite(tensor).all():

        num_nan = torch.isnan(tensor).sum().item()
        num_inf = torch.isinf(tensor).sum().item()

        raise RuntimeError(
            f"\n[SOCIA V2] Non-finite values detected at stage: '{name}'\n"
            f"  shape   : {tuple(tensor.shape)}\n"
            f"  num_nan : {num_nan}\n"
            f"  num_inf : {num_inf}\n"
            f"This pinpoints where the forward pass first breaks down; "
            f"fix the stage immediately preceding '{name}'."
        )


# =====================================================================
# RELATIVE-TIME ATTENTION BIAS
# =====================================================================

class RelativeTimeAttentionBias(nn.Module):
    """
    Computes an additive attention bias from the SIGNED, real elapsed
    time between every pair of events in a window.

    For events i, j: gap_ij = raw_time_i - raw_time_j (hours, signed).
    Positive means event i happened after event j.

    We squash with sign(gap) * log1p(|gap|) for numerical stability
    (raw hour gaps can span from 0 to several thousand), then pass
    through a small MLP that outputs one bias value per attention
    head. The same bias tensor is shared across all encoder layers
    (a deliberate simplification, following the common practice --
    e.g. T5-style relative position biases -- of sharing one
    relative-position signal across layers rather than learning a
    separate one per layer).

    Output shape is (B, nhead, L, L) -- consumed directly by
    ManualSelfAttentionLayer's per-head attention scores, with no
    flattening/reshaping into a (B*nhead, L, L) mask (that flattened
    form was specific to nn.TransformerEncoder's mask argument, which
    is no longer used).
    """

    def __init__(self, nhead: int = NHEAD, hidden_dim: int = REL_TIME_BIAS_HIDDEN_DIM) -> None:
        super().__init__()

        self.nhead = nhead

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nhead),
        )

    def forward(self, raw_delta_hours: torch.Tensor) -> torch.Tensor:
        """
        raw_delta_hours: (B, L) unnormalized hours, non-negative.

        Returns: (B, nhead, L, L) additive float bias.
        """

        t_i = raw_delta_hours.unsqueeze(2)  # (B, L, 1)
        t_j = raw_delta_hours.unsqueeze(1)  # (B, 1, L)

        signed_gap = t_i - t_j  # (B, L, L), signed hours

        scaled_gap = torch.sign(signed_gap) * torch.log1p(torch.abs(signed_gap))

        bias = self.mlp(scaled_gap.unsqueeze(-1))  # (B, L, L, nhead)

        bias = bias.permute(0, 3, 1, 2)  # (B, nhead, L, L)

        return bias


# =====================================================================
# MANUAL SELF-ATTENTION ENCODER LAYER
# =====================================================================

class ManualSelfAttentionLayer(nn.Module):
    """
    Reimplements the same layer architecture as PyTorch's default
    nn.TransformerEncoderLayer (post-norm, ReLU feedforward, dropout
    after attention and after feedforward, plus dropout on attention
    weights) using only plain ops -- no nn.MultiheadAttention, no
    nn.TransformerEncoder, no fused kernels of any kind. This
    guarantees identical numerical behavior in train() and eval()
    modes, which is the property the fused fast-path violated.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:

        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            )

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout1 = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)
        self.resid_dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        check_finite: bool = False,
    ) -> torch.Tensor:
        """
        x         : (B, L, D)
        attn_bias : (B, nhead, L, L) additive bias, already includes
                    all masking/bias information for this layer.
        """

        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        # each: (B, nhead, L, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: (B, nhead, L, L)

        scores = scores + attn_bias

        if check_finite:
            _check_finite(scores, "attention_scores")

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        attn_out = torch.matmul(weights, v)  # (B, nhead, L, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)

        x = x + self.resid_dropout1(attn_out)
        x = self.norm1(x)

        ff = self.linear2(self.ff_dropout(self.activation(self.linear1(x))))
        x = x + self.resid_dropout2(ff)
        x = self.norm2(x)

        if check_finite:
            _check_finite(x, "encoder_layer_output")

        return x


# =====================================================================
# FULL MODEL
# =====================================================================

class SociaPatternTransformerV2(nn.Module):

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

        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=d_model,
        )

        # Projects the 4-dim temporal feature vector (abs time, gap to
        # previous event, gap to previous same-category event, and
        # the has-prior-occurrence flag) into the model dimension.
        self.temporal_projection = nn.Linear(temporal_feature_dim, d_model)

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
        temporal_features  : (B, L, 4) float32, already normalized
                             (see dataset_v2.compute_temporal_features_single)
        raw_delta_hours    : (B, L) float32, UNNORMALIZED hours -- used
                             only to compute the relative-time
                             attention bias, not fed into the token
                             embeddings directly.
        check_finite       : if True, raise immediately with a
                             localized error the first time any
                             intermediate tensor contains NaN/Inf.
                             Off by default (adds negligible cost but
                             is unnecessary once the pipeline is
                             verified); pass True from validation or
                             debugging code.
        """

        if check_finite:
            _check_finite(temporal_features, "temporal_features_input")
            _check_finite(raw_delta_hours, "raw_delta_hours_input")

        category_emb = self.category_embedding(category_ids)  # (B, L, D)

        if check_finite:
            _check_finite(category_emb, "category_embedding")

        temporal_emb = self.temporal_projection(temporal_features)  # (B, L, D)

        if check_finite:
            _check_finite(temporal_emb, "temporal_projection")

        # No positional encoding added here -- see module docstring.
        encoder_input = category_emb + temporal_emb  # (B, L, D)

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