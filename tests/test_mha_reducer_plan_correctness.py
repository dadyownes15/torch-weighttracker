import timm
import pytest
import torch
import torch.nn as nn

from torch_structracker.extractor import FusedQKVExtractor, SeparateQKVExtractor
from torch_structracker.operations import WeightOperationType
from torch_structracker.reducer_plan import (
    compile_reducer_plan_from_groups,
    validate_reducer_plan,
)
from torch_structracker.torch_pruning.dependency import DependencyGraph


class DirectFusedMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(4, 2, batch_first=True, bias=False)

    def forward(self, x):
        output, _ = self.mha(x, x, x, need_weights=False)
        return output


class DirectSeparateMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            4,
            2,
            kdim=6,
            vdim=8,
            batch_first=True,
            bias=False,
        )

    def forward(self, query, key, value):
        output, _ = self.mha(query, key, value, need_weights=False)
        return output


class TinyQKVBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 12, bias=False)
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.proj(q + k + v)


def accumulate_plan(plan):
    output = torch.zeros(plan.output_length)

    for mapping in plan.mappings:
        destination_indices = torch.tensor(
            mapping.destination_indices,
            dtype=torch.long,
        )
        output.index_add_(0, destination_indices, mapping.reducer().reshape(-1))

    return output


def dependency_groups(model, example_inputs, root_module_types):
    graph = DependencyGraph().build_dependency(
        model=model,
        example_inputs=example_inputs,
    )
    return list(graph.get_all_groups(root_module_types=root_module_types))


def qkv_row_reductions(weight, operation_type):
    operation_type = WeightOperationType(operation_type)

    if operation_type == WeightOperationType.SUM:
        return weight.sum(dim=1)
    if operation_type == WeightOperationType.MEAN:
        return weight.mean(dim=1)
    if operation_type == WeightOperationType.COUNT:
        return torch.ones_like(weight).sum(dim=1)
    if operation_type == WeightOperationType.L1:
        return weight.abs().sum(dim=1)
    if operation_type == WeightOperationType.L2:
        return torch.sqrt((weight**2).sum(dim=1))

    raise ValueError(f"Unknown operation type: {operation_type}")


def expected_qkv_output(row_values, embed_dim, num_heads=None, mode=None):
    if mode is None:
        return row_values.reshape(3, embed_dim).sum(dim=0)

    head_dim = embed_dim // num_heads
    values = row_values.reshape(3, num_heads, head_dim)

    if mode == "head_dim":
        return values.sum(dim=(0, 1))
    if mode == "head":
        return values.sum(dim=(0, 2))

    raise ValueError(f"Unknown mode: {mode}")


def fused_qkv_mappings(plan, module):
    return [
        mapping
        for mapping in plan.mappings
        if isinstance(mapping.reducer.parameter_extractor, FusedQKVExtractor)
        and mapping.reducer.parameter_extractor.module is module
    ]


@pytest.mark.parametrize("operation_type", list(WeightOperationType))
def test_direct_fused_mha_plan_matches_exact_raw_qkv_reduction(operation_type):
    model = DirectFusedMHA().eval()
    with torch.no_grad():
        model.mha.in_proj_weight.copy_(torch.arange(1.0, 49.0).reshape(12, 4))
        model.mha.out_proj.weight.zero_()

    groups = dependency_groups(
        model,
        example_inputs=torch.ones(2, 3, 4),
        root_module_types=[nn.MultiheadAttention],
    )
    plan = compile_reducer_plan_from_groups(groups, operation_type=operation_type)

    validate_reducer_plan(plan)
    assert plan.output_length == model.mha.embed_dim
    assert len(plan.mappings) == 1
    assert isinstance(
        plan.mappings[0].reducer.parameter_extractor,
        FusedQKVExtractor,
    )
    assert plan.mappings[0].destination_indices == tuple(range(4)) * 3

    row_values = qkv_row_reductions(model.mha.in_proj_weight, operation_type)
    expected = expected_qkv_output(row_values, embed_dim=model.mha.embed_dim)
    torch.testing.assert_close(accumulate_plan(plan), expected)


def test_direct_fused_mha_prune_dim_plan_matches_exact_head_dim_reduction():
    model = DirectFusedMHA().eval()
    with torch.no_grad():
        model.mha.in_proj_weight.copy_(torch.arange(1.0, 49.0).reshape(12, 4))
        model.mha.out_proj.weight.zero_()

    groups = dependency_groups(
        model,
        example_inputs=torch.ones(2, 3, 4),
        root_module_types=[nn.MultiheadAttention],
    )
    plan = compile_reducer_plan_from_groups(
        groups,
        operation_type=WeightOperationType.SUM,
        num_heads={model.mha: model.mha.num_heads},
        prune_dim=True,
    )

    validate_reducer_plan(plan)
    assert plan.output_length == model.mha.head_dim
    assert len(plan.mappings) == 1
    assert plan.mappings[0].destination_indices == (0, 1, 0, 1) * 3

    row_values = qkv_row_reductions(model.mha.in_proj_weight, WeightOperationType.SUM)
    expected = expected_qkv_output(
        row_values,
        embed_dim=model.mha.embed_dim,
        num_heads=model.mha.num_heads,
        mode="head_dim",
    )
    torch.testing.assert_close(accumulate_plan(plan), expected)


def test_direct_fused_mha_prune_num_heads_plan_matches_exact_head_reduction():
    model = DirectFusedMHA().eval()
    with torch.no_grad():
        model.mha.in_proj_weight.copy_(torch.arange(1.0, 49.0).reshape(12, 4))
        model.mha.out_proj.weight.zero_()

    groups = dependency_groups(
        model,
        example_inputs=torch.ones(2, 3, 4),
        root_module_types=[nn.MultiheadAttention],
    )
    plan = compile_reducer_plan_from_groups(
        groups,
        operation_type=WeightOperationType.SUM,
        num_heads={model.mha: model.mha.num_heads},
        prune_num_heads=True,
    )

    validate_reducer_plan(plan)
    assert plan.output_length == model.mha.num_heads
    assert len(plan.mappings) == 1
    assert plan.mappings[0].destination_indices == (0, 0, 1, 1) * 3

    row_values = qkv_row_reductions(model.mha.in_proj_weight, WeightOperationType.SUM)
    expected = expected_qkv_output(
        row_values,
        embed_dim=model.mha.embed_dim,
        num_heads=model.mha.num_heads,
        mode="head",
    )
    torch.testing.assert_close(accumulate_plan(plan), expected)


@pytest.mark.parametrize("operation_type", list(WeightOperationType))
def test_direct_separate_mha_plan_uses_one_combined_qkv_mapping(operation_type):
    model = DirectSeparateMHA().eval()
    with torch.no_grad():
        model.mha.q_proj_weight.copy_(torch.arange(1.0, 17.0).reshape(4, 4))
        model.mha.k_proj_weight.copy_(torch.arange(17.0, 41.0).reshape(4, 6))
        model.mha.v_proj_weight.copy_(torch.arange(41.0, 73.0).reshape(4, 8))
        model.mha.out_proj.weight.zero_()

    groups = dependency_groups(
        model,
        example_inputs=(
            torch.ones(2, 3, 4),
            torch.ones(2, 3, 6),
            torch.ones(2, 3, 8),
        ),
        root_module_types=[nn.MultiheadAttention],
    )
    plan = compile_reducer_plan_from_groups(
        groups,
        operation_type=operation_type,
    )

    validate_reducer_plan(plan)
    assert plan.output_length == model.mha.embed_dim
    assert len(plan.mappings) == 1
    assert isinstance(
        plan.mappings[0].reducer.parameter_extractor,
        SeparateQKVExtractor,
    )
    assert plan.mappings[0].destination_indices == tuple(range(4)) * 3

    row_values = torch.cat(
        [
            qkv_row_reductions(model.mha.q_proj_weight, operation_type),
            qkv_row_reductions(model.mha.k_proj_weight, operation_type),
            qkv_row_reductions(model.mha.v_proj_weight, operation_type),
        ]
    )
    expected = expected_qkv_output(row_values, embed_dim=model.mha.embed_dim)
    torch.testing.assert_close(accumulate_plan(plan), expected)


def test_fused_qkv_linear_dependency_group_matches_exact_head_dim_reduction():
    model = TinyQKVBlock().eval()
    with torch.no_grad():
        model.qkv.weight.copy_(torch.arange(1.0, 49.0).reshape(12, 4))
        model.proj.weight.zero_()

    groups = dependency_groups(
        model,
        example_inputs=torch.ones(2, 3, 4),
        root_module_types=[nn.Linear],
    )
    qkv_group = next(
        group for group in groups if group[0].dep.target.module is model.qkv
    )
    plan = compile_reducer_plan_from_groups(
        [qkv_group],
        operation_type=WeightOperationType.SUM,
        num_heads={model.qkv: 2},
        prune_dim=True,
    )

    validate_reducer_plan(plan)
    qkv_mappings = fused_qkv_mappings(plan, model.qkv)
    assert plan.output_length == 2
    assert len(qkv_mappings) == 1
    assert len(qkv_mappings[0].destination_indices) == 12
    assert qkv_mappings[0].destination_indices == (0, 1, 0, 1) * 3

    row_values = qkv_row_reductions(model.qkv.weight, WeightOperationType.SUM)
    expected = expected_qkv_output(
        row_values,
        embed_dim=4,
        num_heads=2,
        mode="head_dim",
    )
    torch.testing.assert_close(accumulate_plan(plan), expected)


def test_fused_qkv_linear_dependency_group_matches_exact_head_reduction():
    model = TinyQKVBlock().eval()
    with torch.no_grad():
        model.qkv.weight.copy_(torch.arange(1.0, 49.0).reshape(12, 4))
        model.proj.weight.zero_()

    groups = dependency_groups(
        model,
        example_inputs=torch.ones(2, 3, 4),
        root_module_types=[nn.Linear],
    )
    qkv_group = next(
        group for group in groups if group[0].dep.target.module is model.qkv
    )
    plan = compile_reducer_plan_from_groups(
        [qkv_group],
        operation_type=WeightOperationType.SUM,
        num_heads={model.qkv: 2},
        prune_num_heads=True,
    )

    validate_reducer_plan(plan)
    qkv_mappings = fused_qkv_mappings(plan, model.qkv)
    assert plan.output_length == 2
    assert len(qkv_mappings) == 1
    assert len(qkv_mappings[0].destination_indices) == 12
    assert qkv_mappings[0].destination_indices == (0, 0, 1, 1) * 3

    row_values = qkv_row_reductions(model.qkv.weight, WeightOperationType.SUM)
    expected = expected_qkv_output(
        row_values,
        embed_dim=4,
        num_heads=2,
        mode="head",
    )
    torch.testing.assert_close(accumulate_plan(plan), expected)


def test_real_timm_vit_qkv_plan_uses_single_fused_qkv_reducer_mapping():
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        img_size=32,
        num_classes=0,
    ).eval()
    attention = model.blocks[0].attn
    groups = dependency_groups(
        model,
        example_inputs=torch.ones(1, 3, 32, 32),
        root_module_types=[nn.Linear],
    )
    qkv_group = next(
        group for group in groups if group[0].dep.target.module is attention.qkv
    )

    plan = compile_reducer_plan_from_groups(
        [qkv_group],
        operation_type=WeightOperationType.SUM,
        num_heads={attention.qkv: attention.num_heads},
        prune_num_heads=True,
    )

    validate_reducer_plan(plan)
    qkv_mappings = fused_qkv_mappings(plan, attention.qkv)
    assert plan.output_length == attention.num_heads
    assert len(qkv_mappings) == 1
    assert qkv_mappings[0].reducer().numel() == attention.qkv.out_features
    assert len(qkv_mappings[0].destination_indices) == attention.qkv.out_features
    assert set(qkv_mappings[0].destination_indices) == set(
        range(attention.num_heads)
    )
