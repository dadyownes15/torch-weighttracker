from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
from torch import Tensor

import torch_structure_analyser as tsa
from tests.fixtures_models import TinyTransformerClassifier


class TinyResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + identity)


class TinyResNetClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(8)
        self.stem_act = nn.ReLU()
        self.block1 = TinyResNetBlock(8, 8)
        self.block2 = TinyResNetBlock(8, 16, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(16, 5)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

@dataclass(frozen=True)
class StructuredZeroCase:
    group_id: str
    zero_mode: str
    unit_root_indices: tuple[int, ...]


def _make_resnet_controller(model: TinyResNetClassifier | None = None) -> tsa.SparsityTracker:
    if model is None:
        model = TinyResNetClassifier()
    return tsa.SparsityTracker(
        model,
        example_inputs=torch.randn(1, 3, 16, 16),
        verbose=False,
    )


def _make_vit_controller(model: TinyTransformerClassifier | None = None) -> tsa.SparsityTracker:
    if model is None:
        model = TinyTransformerClassifier()
    return tsa.SparsityTracker(
        model,
        example_inputs=torch.randint(0, 32, (1, 8)),
        root_module_types=[nn.MultiheadAttention, nn.Linear],
        num_heads={model.attn: model.attn.num_heads},
        prune_num_heads=True,
        prune_head_dims=True,
        verbose=False,
    )


def test_tiny_vit_structure_summary_shows_attention_coupling():
    controller = _make_vit_controller()

    summary = controller.format_structure_summary()

    assert "attn:prune_out_channels:head" in summary
    assert "  members for attn:prune_out_channels:head:" in summary
    assert "    - norm1" in summary
    assert "    - attn" in summary
    assert "    - mlp_in" in summary
    assert "    - mlp_out" in summary


def test_tiny_resnet_structure_summary_shows_residual_coupling():
    controller = _make_resnet_controller()

    summary = controller.format_structure_summary()

    assert "block2.conv1:prune_out_channels" in summary
    assert "\n  members for block2.conv1:prune_out_channels:\n" in summary
    assert "block2.conv1" in summary
    assert "block2.bn1" in summary
    assert "block2.conv2" in summary


def _zero_mha_slices(attention: nn.MultiheadAttention, idxs: list[int], include_bias: bool) -> None:
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
        if include_bias and attention.in_proj_bias is not None:
            attention.in_proj_bias[repeated_idxs] = 0
        if attention.bias_k is not None:
            attention.bias_k[:, :, idxs] = 0
        if attention.bias_v is not None:
            attention.bias_v[:, :, idxs] = 0
        if attention.out_proj is not None:
            attention.out_proj.weight[idxs, :] = 0
            attention.out_proj.weight[:, idxs] = 0
            if include_bias and attention.out_proj.bias is not None:
                attention.out_proj.bias[idxs] = 0


def _zero_member_slices(module, handler, idxs: list[int], include_bias: bool) -> None:
    with torch.no_grad():
        if handler in [tsa.prune_conv_out_channels, tsa.prune_depthwise_conv_out_channels]:
            module.weight[idxs, ...] = 0
            if include_bias and module.bias is not None:
                module.bias[idxs] = 0
        elif handler in [tsa.prune_conv_in_channels, tsa.prune_depthwise_conv_in_channels]:
            module.weight[:, idxs, ...] = 0
        elif handler in [tsa.prune_linear_out_channels]:
            module.weight[idxs, :] = 0
            if include_bias and module.bias is not None:
                module.bias[idxs] = 0
        elif handler in [tsa.prune_linear_in_channels]:
            module.weight[:, idxs] = 0
        elif handler in [
            tsa.prune_batchnorm_out_channels,
            tsa.prune_batchnorm_in_channels,
            tsa.prune_groupnorm_out_channels,
            tsa.prune_groupnorm_in_channels,
            tsa.prune_instancenorm_out_channels,
            tsa.prune_instancenorm_in_channels,
        ]:
            if getattr(module, "affine", False):
                module.weight[idxs] = 0
                if include_bias and module.bias is not None:
                    module.bias[idxs] = 0
        elif handler in [tsa.prune_layernorm_out_channels, tsa.prune_layernorm_in_channels]:
            if getattr(module, "elementwise_affine", False):
                module.weight[idxs] = 0
                if include_bias and module.bias is not None:
                    module.bias[idxs] = 0
        elif handler in [tsa.prune_embedding_out_channels, tsa.prune_embedding_in_channels]:
            module.weight[:, idxs] = 0
        elif handler in [tsa.prune_multihead_attention_out_channels, tsa.prune_multihead_attention_in_channels]:
            _zero_mha_slices(module, idxs, include_bias=include_bias)
        else:
            raise AssertionError(f"Unhandled test zeroing rule for {handler.__name__}")


def _zero_group_unit(
    controller: tsa.SparsityTracker,
    group_id: str,
    unit_root_indices: tuple[int, ...],
    *,
    zero_mode: str,
    include_bias: bool,
) -> None:
    group_view = next(view for view in controller.iter_groups() if view.group_id == group_id)
    members = group_view.members[:1] if zero_mode == "root_only" else group_view.members
    root_set = set(unit_root_indices)

    for member in members:
        if not member.measurable:
            continue
        local_idxs = [
            local_idx
            for local_idx, root_idx in zip(member.local_idxs, member.root_idxs)
            if root_idx in root_set
        ]
        if len(local_idxs) == 0:
            continue
        _zero_member_slices(member.module, member.handler, local_idxs, include_bias=include_bias)


def _candidate_map(candidates) -> dict[str, object]:
    return {candidate.group_id: candidate for candidate in candidates}


@pytest.mark.parametrize(
    "case",
    [
        StructuredZeroCase(
            group_id="block1.conv1:prune_out_channels",
            zero_mode="root_only",
            unit_root_indices=(0,),
        ),
        StructuredZeroCase(
            group_id="block1.conv1:prune_out_channels",
            zero_mode="group",
            unit_root_indices=(0,),
        ),
        StructuredZeroCase(
            group_id="block2.conv1:prune_out_channels",
            zero_mode="root_only",
            unit_root_indices=(0,),
        ),
        StructuredZeroCase(
            group_id="block2.conv1:prune_out_channels",
            zero_mode="group",
            unit_root_indices=(0,),
        ),
        StructuredZeroCase(
            group_id="block2.downsample.0:prune_out_channels",
            zero_mode="root_only",
            unit_root_indices=(0,),
        ),
        StructuredZeroCase(
            group_id="block2.downsample.0:prune_out_channels",
            zero_mode="group",
            unit_root_indices=(0,),
        ),
    ],
)
def test_resnet_structured_sparsity_requires_zeroing_the_full_dependency_unit(case: StructuredZeroCase):
    controller = _make_resnet_controller()

    _zero_group_unit(
        controller,
        case.group_id,
        case.unit_root_indices,
        zero_mode=case.zero_mode,
        include_bias=True,
    )

    report = controller.structured_sparsity(include_bias=True)
    candidates = _candidate_map(controller.zero_structure_candidates(include_bias=True))
    group_stats = report.by_group[case.group_id]

    expected_zero_units = (case.unit_root_indices,) if case.zero_mode == "group" else ()
    assert group_stats.zero_prune_units == expected_zero_units
    assert group_stats.stats.removed == (1 if case.zero_mode == "group" else 0)

    if case.zero_mode == "group":
        assert candidates[case.group_id].zero_prune_units == expected_zero_units
    else:
        assert case.group_id not in candidates


@pytest.mark.parametrize(
    "case",
    [
        StructuredZeroCase(
            group_id="mlp_in:prune_out_channels",
            zero_mode="root_only",
            unit_root_indices=(0,),
        ),
        StructuredZeroCase(
            group_id="mlp_in:prune_out_channels",
            zero_mode="group",
            unit_root_indices=(0,),
        ),
        StructuredZeroCase(
            group_id="attn:prune_out_channels:head_dim",
            zero_mode="root_only",
            unit_root_indices=(0, 4, 8, 12),
        ),
        StructuredZeroCase(
            group_id="attn:prune_out_channels:head_dim",
            zero_mode="group",
            unit_root_indices=(0, 4, 8, 12),
        ),
        StructuredZeroCase(
            group_id="attn:prune_out_channels:head",
            zero_mode="root_only",
            unit_root_indices=(0, 1, 2, 3),
        ),
        StructuredZeroCase(
            group_id="attn:prune_out_channels:head",
            zero_mode="group",
            unit_root_indices=(0, 1, 2, 3),
        ),
    ],
)
def test_vit_structured_sparsity_requires_zeroing_the_full_dependency_unit(case: StructuredZeroCase):
    controller = _make_vit_controller()

    _zero_group_unit(
        controller,
        case.group_id,
        case.unit_root_indices,
        zero_mode=case.zero_mode,
        include_bias=True,
    )

    report = controller.structured_sparsity(include_bias=True)
    candidates = _candidate_map(controller.zero_structure_candidates(include_bias=True))
    group_stats = report.by_group[case.group_id]

    expected_zero_units = (case.unit_root_indices,) if case.zero_mode == "group" else ()
    assert group_stats.zero_prune_units == expected_zero_units
    assert group_stats.stats.removed == (1 if case.zero_mode == "group" else 0)

    if case.zero_mode == "group":
        assert candidates[case.group_id].zero_prune_units == expected_zero_units
    else:
        assert case.group_id not in candidates


def test_resnet_prune_zero_structures_prunes_a_zeroed_residual_unit():
    model = TinyResNetClassifier()
    controller = _make_resnet_controller(model)
    reference = controller.capture_reference()

    _zero_group_unit(
        controller,
        "block2.downsample.0:prune_out_channels",
        (0,),
        zero_mode="group",
        include_bias=True,
    )

    result = controller.prune_zero_structures(include_bias=True)
    report = controller.structured_sparsity(reference=reference, include_bias=True)

    assert "block2.downsample.0:prune_out_channels" in result.pruned_group_ids
    assert model.block2.conv2.out_channels == 15
    assert model.block2.downsample[0].out_channels == 15
    assert model.head.in_features == 15
    assert report.by_group["block2.downsample.0:prune_out_channels"].stats.removed == 1


def test_vit_prune_zero_structures_prunes_a_zeroed_attention_head():
    model = TinyTransformerClassifier()
    controller = _make_vit_controller(model)

    _zero_group_unit(
        controller,
        "attn:prune_out_channels:head",
        (0, 1, 2, 3),
        zero_mode="group",
        include_bias=True,
    )

    result = controller.prune_zero_structures(include_bias=True)
    attention_views = [view for view in controller.iter_groups() if view.axis == tsa.StructureAxis.HEAD]

    assert result.pruned_group_ids == ("attn:prune_out_channels:head",)
    assert controller.config.num_heads[model.attn] == 3
    assert attention_views[0].size == 3
