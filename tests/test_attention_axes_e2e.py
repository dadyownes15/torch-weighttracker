from __future__ import annotations

import torch
import torch.nn as nn

import torch_structure_analyser as tsa
from torch_structure_analyser.analysis import StructureAxis
from tests.fixtures_models import TinyTransformerClassifier


class AttentionProbeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
        self.lin = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        y, _ = self.mha(x, x, x)
        return self.lin(y)


def _make_probe_controller(model: AttentionProbeNet | None = None) -> tsa.SparsityTracker:
    if model is None:
        model = AttentionProbeNet()
    return tsa.SparsityTracker(
        model,
        example_inputs=torch.randn(1, 4, 8),
        root_module_types=[nn.MultiheadAttention, nn.Linear],
        num_heads={model.mha: 2},
        prune_num_heads=True,
        prune_head_dims=True,
    )


def _make_transformer_controller(model: TinyTransformerClassifier | None = None) -> tsa.SparsityTracker:
    if model is None:
        model = TinyTransformerClassifier()
    token_ids = torch.randint(0, 32, (1, 8))
    return tsa.SparsityTracker(
        model,
        example_inputs=token_ids,
        root_module_types=[nn.MultiheadAttention, nn.Linear],
        num_heads={model.attn: model.attn.num_heads},
        prune_num_heads=True,
        prune_head_dims=True,
    )


def _zero_mha_out_slices(attention: nn.MultiheadAttention, idxs: list[int]) -> None:
    embed_dim = attention.embed_dim
    repeated_idxs = idxs + [idx + embed_dim for idx in idxs] + [idx + 2 * embed_dim for idx in idxs]
    with torch.no_grad():
        if attention.q_proj_weight is not None:
            attention.q_proj_weight[idxs, :] = 0
        if attention.k_proj_weight is not None:
            attention.k_proj_weight[idxs, :] = 0
        if attention.v_proj_weight is not None:
            attention.v_proj_weight[idxs, :] = 0
        if attention.in_proj_weight is not None:
            attention.in_proj_weight[repeated_idxs, :] = 0
            attention.in_proj_weight[:, idxs] = 0
        if attention.out_proj is not None:
            attention.out_proj.weight[idxs, :] = 0
            attention.out_proj.weight[:, idxs] = 0


def _zero_member_slices(module, handler, idxs: list[int]) -> None:
    with torch.no_grad():
        if handler in [tsa.prune_linear_out_channels]:
            module.weight[idxs, :] = 0
            if module.bias is not None:
                module.bias[idxs] = 0
        elif handler in [tsa.prune_linear_in_channels]:
            module.weight[:, idxs] = 0
        elif handler in [tsa.prune_layernorm_out_channels]:
            if module.elementwise_affine:
                module.weight[idxs] = 0
                if module.bias is not None:
                    module.bias[idxs] = 0
        elif handler in [tsa.prune_batchnorm_out_channels, tsa.prune_groupnorm_out_channels, tsa.prune_instancenorm_out_channels]:
            if getattr(module, "affine", False):
                module.weight[idxs] = 0
                if module.bias is not None:
                    module.bias[idxs] = 0
        elif handler in [tsa.prune_embedding_out_channels]:
            module.weight[:, idxs] = 0
        elif handler in [tsa.prune_multihead_attention_out_channels]:
            _zero_mha_out_slices(module, idxs)
        else:
            raise AssertionError(f"Unhandled test zeroing rule for {handler.__name__}")


def _zero_group_unit(controller: tsa.SparsityTracker, group_id: str, root_indices: list[int]) -> None:
    group_view = next(view for view in controller.iter_groups() if view.group_id == group_id)
    root_set = set(root_indices)
    for member in group_view.members:
        if not member.measurable:
            continue
        local_idxs = [
            local_idx
            for local_idx, root_idx in zip(member.local_idxs, member.root_idxs)
            if root_idx in root_set
        ]
        if len(local_idxs) == 0:
            continue
        _zero_member_slices(member.module, member.handler, local_idxs)


def test_attention_head_dim_report_detects_zeroed_shared_dimension():
    controller = _make_probe_controller()
    _zero_group_unit(controller, "lin:prune_in_channels:head_dim", [0, 4])

    report = controller.structured_sparsity()

    assert report.by_group["lin:prune_in_channels:head_dim"].zero_prune_units == ((0, 4),)
    assert report.by_group["lin:prune_in_channels:head"].stats.removed == 0


def test_attention_head_report_detects_zeroed_whole_head():
    controller = _make_probe_controller()
    _zero_group_unit(controller, "lin:prune_in_channels:head", [0, 1, 2, 3])

    report = controller.structured_sparsity()

    assert report.by_group["lin:prune_in_channels:head"].zero_prune_units == ((0, 1, 2, 3),)
    assert report.by_group["lin:prune_in_channels:head_dim"].stats.removed == 0


def test_prune_zero_structures_updates_logical_num_heads_after_head_prune():
    model = TinyTransformerClassifier()
    controller = _make_transformer_controller(model)
    _zero_group_unit(controller, "attn:prune_out_channels:head", [0, 1, 2, 3])

    result = controller.prune_zero_structures()
    attention_views = [view for view in controller.iter_groups() if view.axis == StructureAxis.HEAD]

    assert result.pruned_group_ids == ("attn:prune_out_channels:head",)
    assert controller.config.num_heads[model.attn] == 3
    assert attention_views[0].size == 3


def test_tiny_transformer_exposes_attention_head_and_head_dim_views_end_to_end():
    controller = _make_transformer_controller()

    attention_views = [view for view in controller.iter_groups() if view.group_id.startswith("attn:prune_out_channels")]

    assert len(attention_views) == 2
    assert {view.axis for view in attention_views} == {StructureAxis.HEAD, StructureAxis.HEAD_DIM}
