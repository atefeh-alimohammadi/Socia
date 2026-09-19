from __future__ import annotations

import torch
import torch.nn as nn

from dataset import build_datasets, create_dataloader
from model import SociaPatternTransformer, D_MODEL, WINDOW_LENGTH


def sanity_check(batch_size: int = 64) -> None:

    (
        train_dataset,
        _val_dataset,
        _test_dataset,
        _time_stats,
    ) = build_datasets()

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    category_ids, delta_hours, labels = next(iter(train_loader))

    model = SociaPatternTransformer()

    print()
    print("=" * 78)
    print("SOCIA MODEL SANITY CHECK")
    print("=" * 78)

    print()
    print("Input shapes:")
    print(f"  category_ids : {category_ids.shape}, {category_ids.dtype}")
    print(f"  delta_hours  : {delta_hours.shape}, {delta_hours.dtype}")
    print(f"  labels       : {labels.shape}, {labels.dtype}")

    logits, attention_weights, pooled = model(
        category_ids,
        delta_hours,
        return_attention=True,
    )

    print()
    print("Output shapes:")
    print(f"  logits            : {logits.shape}")
    print(f"  attention_weights : {attention_weights.shape}")
    print(f"  pooled            : {pooled.shape}")

    expected_batch = min(batch_size, len(train_dataset))

    assert logits.shape == (expected_batch,), (
        f"Expected logits shape ({expected_batch},), got {logits.shape}"
    )

    assert attention_weights.shape == (expected_batch, WINDOW_LENGTH), (
        f"Expected attention_weights shape ({expected_batch}, {WINDOW_LENGTH}), "
        f"got {attention_weights.shape}"
    )

    assert pooled.shape == (expected_batch, D_MODEL), (
        f"Expected pooled shape ({expected_batch}, {D_MODEL}), got {pooled.shape}"
    )

    logits_finite = torch.isfinite(logits).all().item()
    attn_finite = torch.isfinite(attention_weights).all().item()
    pooled_finite = torch.isfinite(pooled).all().item()

    assert logits_finite, "logits contain NaN/Inf"
    assert attn_finite, "attention_weights contain NaN/Inf"
    assert pooled_finite, "pooled contains NaN/Inf"

    attn_sums = attention_weights.sum(dim=1)
    attn_sums_ok = torch.allclose(
        attn_sums,
        torch.ones_like(attn_sums),
        atol=1e-5,
    )

    assert attn_sums_ok, "attention_weights do not sum to 1 per row"

    print()
    print("Finite checks:")
    print(f"  logits NaN/Inf            : {not logits_finite}")
    print(f"  attention_weights NaN/Inf : {not attn_finite}")
    print(f"  pooled NaN/Inf            : {not pooled_finite}")
    print(f"  attention rows sum to 1   : {bool(attn_sums_ok)}")

    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, labels)

    print()
    print(f"BCEWithLogitsLoss value: {loss.item():.6f}")

    assert torch.isfinite(loss).item(), "loss is not finite"

    model.zero_grad()
    loss.backward()

    total_params = 0
    params_with_grad = 0
    params_with_nonzero_grad = 0

    for name, param in model.named_parameters():

        total_params += 1

        if param.grad is not None:

            params_with_grad += 1

            if torch.any(param.grad != 0):
                params_with_nonzero_grad += 1
            else:
                print(f"  WARNING: zero gradient for '{name}'")

        else:
            print(f"  WARNING: no gradient for '{name}'")

    print()
    print("Backward pass:")
    print(f"  total parameters          : {total_params}")
    print(f"  parameters with grad      : {params_with_grad}")
    print(f"  parameters with nonzero grad : {params_with_nonzero_grad}")

    assert params_with_grad == total_params, (
        "Some parameters received no gradient at all"
    )

    assert params_with_nonzero_grad == total_params, (
        "Some parameters received an all-zero gradient"
    )

    print()
    print("=" * 78)
    print("STEP 2 SANITY CHECK: PASS")
    print("=" * 78)
    print()


if __name__ == "__main__":
    sanity_check()