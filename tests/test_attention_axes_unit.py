from __future__ import annotations

import torch
import torch.nn as nn

import torch_structure_analyser as tsa
from torch_structure_analyser.analysis import (
    StructureAxis,
    build_atomic_prune_units,
    build_grouped_prune_units,
    build_head_prune_units,
)


class AttentionProbeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
        self.lin = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        y, _ = self.mha(x, x, x)
        return self.lin(y)


def _make_controller() -> tsa.SparsityTracker:
    model = AttentionProbeNet()
    return tsa.SparsityTracker(
        model,
        example_inputs=torch.randn(1, 4, 8),
        root_module_types=[nn.MultiheadAttention, nn.Linear],
        num_heads={model.mha: 2},
        prune_num_heads=True,
        prune_head_dims=True,
    )


def _zero_attention_indices(model: AttentionProbeNet, idxs: list[int]) -> None:
    embed_dim = model.mha.embed_dim
    repeated_idxs = idxs + [idx + embed_dim for idx in idxs] + [idx + 2 * embed_dim for idx in idxs]
    with torch.no_grad():
        if model.mha.q_proj_weight is not None:
            model.mha.q_proj_weight[idxs, :] = 0
        if model.mha.k_proj_weight is not None:
            model.mha.k_proj_weight[idxs, :] = 0
        if model.mha.v_proj_weight is not None:
            model.mha.v_proj_weight[idxs, :] = 0
        if model.mha.in_proj_weight is not None:
            model.mha.in_proj_weight[repeated_idxs, :] = 0
            model.mha.in_proj_weight[:, idxs] = 0
        if model.mha.out_proj is not None:
            model.mha.out_proj.weight[idxs, :] = 0
            model.mha.out_proj.weight[:, idxs] = 0
        model.lin.weight[:, idxs] = 0


def test_build_atomic_prune_units_returns_singletons():
    units = build_atomic_prune_units(4)
    assert [unit.root_indices for unit in units] == [(0,), (1,), (2,), (3,)]


def test_build_grouped_prune_units_matches_head_dim_pattern():
    units = build_grouped_prune_units(8, 2)
    assert [unit.root_indices for unit in units] == [(0, 4), (1, 5), (2, 6), (3, 7)]


def test_build_head_prune_units_returns_contiguous_blocks():
    units = build_head_prune_units(8, 2)
    assert [unit.root_indices for unit in units] == [(0, 1, 2, 3), (4, 5, 6, 7)]


def test_attention_group_views_include_head_and_head_dim_axes():
    controller = _make_controller()
    attention_views = [view for view in controller.iter_groups() if view.group_id.startswith("lin:prune_in_channels")]

    assert [view.axis for view in attention_views] == [StructureAxis.HEAD_DIM, StructureAxis.HEAD]
    assert [unit.root_indices for unit in attention_views[0].prune_units] == [(0, 4), (1, 5), (2, 6), (3, 7)]
    assert [unit.root_indices for unit in attention_views[1].prune_units] == [(0, 1, 2, 3), (4, 5, 6, 7)]


def test_zero_structure_candidates_prioritize_heads_before_head_dims():
    controller = _make_controller()
    _zero_attention_indices(controller.model, list(range(8)))

    candidates = controller.zero_structure_candidates()
    attention_candidates = [candidate for candidate in candidates if candidate.group_id.startswith("lin:prune_in_channels")]

    assert [candidate.axis for candidate in attention_candidates[:2]] == [StructureAxis.HEAD, StructureAxis.HEAD_DIM]


def test_group_lasso_axes_filter_keeps_attention_views_separate():
    controller = _make_controller()

    head_loss, head_named_terms = controller.group_lasso(axes=(StructureAxis.HEAD,))
    head_dim_loss, head_dim_named_terms = controller.group_lasso(axes=(StructureAxis.HEAD_DIM,))

    assert head_loss.ndim == 0
    assert head_dim_loss.ndim == 0
    assert head_loss.requires_grad
    assert head_dim_loss.requires_grad
    assert head_named_terms
    assert head_dim_named_terms
    assert all("[0]" in key or "[1]" in key for key in head_named_terms)
    assert len(head_named_terms) < len(head_dim_named_terms)
