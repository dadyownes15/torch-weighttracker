import pytest
import torch

from torch_weighttracker import WeightTracker
from torch_weighttracker.canonical_units import UnitKind
from torch_weighttracker.integrations.timm import (
    infer_vit_num_heads,
    sync_vit_attention_metadata,
    vit_pruning_config,
)


def test_timm_vit_prune_head_syncs_attention_metadata_and_recreates_tracker() -> None:
    pytest.importorskip("timm")
    from timm.models.vision_transformer import VisionTransformer

    model = VisionTransformer(
        img_size=16,
        patch_size=8,
        embed_dim=24,
        depth=1,
        num_heads=3,
        mlp_ratio=1.0,
        num_classes=2,
    )
    example_inputs = torch.randn(1, 3, 16, 16)
    config = vit_pruning_config(model)
    tracker = WeightTracker(
        model,
        example_inputs=example_inputs,
        prune_num_heads=True,
        **config,
    )

    qkv_to_heads = infer_vit_num_heads(model)
    assert len(qkv_to_heads) == 1
    qkv = next(iter(qkv_to_heads))
    group_id = next(
        group_id
        for group_id, group in enumerate(tracker.canonical_groups)
        if group.unit_kind == UnitKind.HEAD
        and any(member.module is qkv for member in group.members)
    )

    result = tracker.prune_unit(group_id, 0)
    attention = model.blocks[0].attn

    assert len(result.pruning_idxs) == 24
    assert qkv.out_features == 48
    assert tracker.num_heads[qkv] == 2
    assert attention.num_heads == 2
    assert attention.attn_dim == 16
    assert attention.head_dim == 8
    assert attention.scale == 8**-0.5
    assert model(example_inputs).shape == (1, 2)

    metrics = tracker.create_tracker("structured_bops", log_total_bops=True).track()
    assert "structured_bops" in metrics


def test_timm_vit_direct_head_prune_does_not_track_projection_as_qkv() -> None:
    pytest.importorskip("timm")
    from timm.models.vision_transformer import VisionTransformer

    model = VisionTransformer(
        img_size=16,
        patch_size=8,
        embed_dim=24,
        depth=1,
        num_heads=3,
        mlp_ratio=1.0,
        num_classes=2,
    )
    example_inputs = torch.randn(1, 3, 16, 16)
    qkv_to_heads = infer_vit_num_heads(model)
    qkv = next(iter(qkv_to_heads))
    attention = model.blocks[0].attn
    tracker = WeightTracker(
        model,
        example_inputs=example_inputs,
        num_heads=qkv_to_heads,
        prune_num_heads=True,
        post_prune_hooks=(sync_vit_attention_metadata,),
    )

    result = tracker.prune_unit(group_id=3, unit_id=1)

    assert len(result.pruning_idxs) == 24
    assert qkv.out_features == 48
    assert tracker.num_heads[qkv] == 2
    assert attention.proj not in tracker.num_heads
    assert attention.num_heads == 2
    assert attention.attn_dim == 16
    assert attention.head_dim == 8
    assert model(example_inputs).shape == (1, 2)
