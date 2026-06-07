from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from torch_weighttracker import WeightTracker
from torch_weighttracker.canonical_units import (
    PruningIndexLayout,
    UnitKind,
    canonicalize_groups,
)
from torch_weighttracker.pruning.fake import fake_prune_canonical_unit
from torch_weighttracker.torch_pruning.pruner.function import (
    prune_linear_out_channels,
    prune_multihead_attention_out_channels,
)


class FakeGroup:
    def __init__(self, *items) -> None:
        self.items = list(items)


def _member(module: nn.Module, handler, indices: tuple[int, ...]):
    return SimpleNamespace(
        dep=SimpleNamespace(
            target=SimpleNamespace(module=module),
            handler=handler,
        ),
        root_idxs=indices,
        idxs=indices,
    )


class TinyMha(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=24,
            num_heads=3,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.mha(x, x, x)
        return y


class TinyMhaChain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(24, 24)
        self.mha = nn.MultiheadAttention(
            embed_dim=24,
            num_heads=3,
            batch_first=True,
        )
        self.post = nn.Linear(24, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        y, _ = self.mha(x, x, x)
        return self.post(y)


def _head_group_id(tracker: WeightTracker, module: nn.Module) -> int:
    return next(
        group_id
        for group_id, group in enumerate(tracker.canonical_groups)
        if group.unit_kind == UnitKind.HEAD
        and any(member.module is module for member in group.members)
    )


def test_mha_head_pruning_indices_use_embed_space() -> None:
    model = TinyMha()
    groups = canonicalize_groups(
        (
            FakeGroup(
                _member(
                    model.mha,
                    prune_multihead_attention_out_channels,
                    tuple(range(24)),
                )
            ),
        ),
        num_heads={model.mha: 3},
        prune_num_heads=True,
    )

    member = groups[0].members[0]

    assert member.projection_out_features == 24
    assert member.projection_in_features == 24
    assert member.embed_dim == 24
    assert member.pruning_index_layout == PruningIndexLayout.EMBED_SPACE
    assert member.pruning_indices_by_unit[0] == tuple(range(8))


def test_fused_qkv_head_pruning_indices_use_repeated_row_space() -> None:
    qkv = nn.Linear(16, 72, bias=False)
    groups = canonicalize_groups(
        (
            FakeGroup(
                _member(qkv, prune_linear_out_channels, tuple(range(72))),
            ),
        ),
        num_heads={qkv: 3},
        prune_num_heads=True,
    )

    member = groups[0].members[0]

    assert member.projection_out_features == 24
    assert member.projection_in_features == 16
    assert member.embed_dim == 24
    assert member.pruning_index_layout == PruningIndexLayout.FUSED_QKV_ROW_SPACE
    assert member.pruning_indices_by_unit[0] == (
        *range(8),
        *range(24, 32),
        *range(48, 56),
    )


def test_attention_head_grouping_uses_projection_out_features() -> None:
    qkv = nn.Linear(16, 72, bias=False)
    groups = canonicalize_groups(
        (
            FakeGroup(
                _member(qkv, prune_linear_out_channels, tuple(range(72))),
            ),
        ),
        num_heads={qkv: 3},
        prune_num_heads=True,
    )

    assert groups[0].length == 3
    assert groups[0].members[0].head_dim == 8


def test_physical_mha_head_prune_updates_num_heads_before_torch_pruning() -> None:
    model = TinyMhaChain()
    example_inputs = torch.randn(1, 4, 24)
    tracker = WeightTracker(
        model,
        example_inputs=example_inputs,
        root_module_types=[nn.MultiheadAttention, nn.Linear],
        num_heads={model.mha: 3},
        prune_num_heads=True,
    )

    result = tracker.prune_unit(_head_group_id(tracker, model.mha), 0)

    assert result.pruning_idxs == tuple(range(8))
    assert model.mha.embed_dim == 16
    assert model.mha.num_heads == 2
    assert model.mha.head_dim == 8
    assert tracker.num_heads[model.mha] == 2
    assert model(example_inputs).shape == (1, 4, 2)


def test_fake_pruning_mha_head_expands_embed_indices_to_qkv_rows() -> None:
    model = TinyMha()
    groups = canonicalize_groups(
        (
            FakeGroup(
                _member(
                    model.mha,
                    prune_multihead_attention_out_channels,
                    tuple(range(24)),
                )
            ),
        ),
        num_heads={model.mha: 3},
        prune_num_heads=True,
    )

    fake_prune_canonical_unit(model, groups, 0, 0)

    rows = (*range(8), *range(24, 32), *range(48, 56))
    torch.testing.assert_close(
        model.mha.in_proj_weight[list(rows)],
        torch.zeros_like(model.mha.in_proj_weight[list(rows)]),
    )
    assert not torch.allclose(
        model.mha.in_proj_weight[list(range(8, 24))],
        torch.zeros_like(model.mha.in_proj_weight[list(range(8, 24))]),
    )


def test_fake_pruning_fused_qkv_head_uses_repeated_linear_rows() -> None:
    model = nn.Sequential(nn.Linear(16, 72, bias=False))
    qkv = model[0]
    groups = canonicalize_groups(
        (
            FakeGroup(
                _member(qkv, prune_linear_out_channels, tuple(range(72))),
            ),
        ),
        num_heads={qkv: 3},
        prune_num_heads=True,
    )

    fake_prune_canonical_unit(model, groups, 0, 0)

    rows = (*range(8), *range(24, 32), *range(48, 56))
    torch.testing.assert_close(
        qkv.weight[list(rows)],
        torch.zeros_like(qkv.weight[list(rows)]),
    )
    assert not torch.allclose(
        qkv.weight[list(range(8, 24))],
        torch.zeros_like(qkv.weight[list(range(8, 24))]),
    )
