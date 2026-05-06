import timm
import pytest
import torch
import torch.nn as nn

from torch_structracker.operations import (
    FusedQKVHeadOperation,
    WeightOperationType,
)
from torch_structracker.reducer_plan import (
    compile_reducer_plan_from_groups,
    compile_reducer_plan_from_modules,
    validate_reducer_plan,
)
from torch_structracker.torch_pruning.dependency import DependencyGraph


def make_tiny_vit():
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        img_size=32,
        num_classes=0,
    )
    model.eval()
    return model


def qkv_dependency_group(model, attention):
    graph = DependencyGraph().build_dependency(
        model=model,
        example_inputs=torch.ones(1, 3, 32, 32),
    )
    groups = list(graph.get_all_groups(root_module_types=[nn.Linear]))
    return next(group for group in groups if group[0].dep.target.module is attention.qkv)


def test_timm_vit_module_plan_sums_real_model_weights():
    model = make_tiny_vit()

    plan = compile_reducer_plan_from_modules(
        model,
        operation_type=WeightOperationType.SUM,
    )

    validate_reducer_plan(plan)
    assert "patch_embed.proj" in plan.output_labels
    assert "blocks.0.norm1" in plan.output_labels
    assert "blocks.0.attn.qkv" in plan.output_labels
    assert "blocks.0.attn.proj" in plan.output_labels


def test_timm_vit_qkv_head_operation_reduces_real_attention_weight():
    model = make_tiny_vit()
    attention = model.blocks[0].attn
    operation = FusedQKVHeadOperation(
        operation_type=WeightOperationType.SUM,
        num_heads=attention.num_heads,
        head_dim=attention.head_dim,
    )

    result = operation(attention.qkv.weight)

    expected = (
        attention.qkv.weight.sum(dim=1)
        .reshape(3, attention.num_heads, attention.head_dim)
        .sum(dim=(0, 2))
    )
    assert result.shape == (attention.num_heads,)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("operation_type", list(WeightOperationType))
def test_timm_vit_linear_dependency_groups_compile_structured_plan_for_all_operations(
    operation_type,
):
    model = make_tiny_vit()
    graph = DependencyGraph().build_dependency(
        model=model,
        example_inputs=torch.ones(1, 3, 32, 32),
    )
    groups = list(graph.get_all_groups(root_module_types=[nn.Linear]))
    module_names_by_id = {
        id(module): name for name, module in model.named_modules()
    }
    root_names = {
        module_names_by_id.get(id(group[0].dep.target.module)) for group in groups
    }

    assert "blocks.0.attn.qkv" in root_names

    plan = compile_reducer_plan_from_groups(
        groups,
        operation_type=operation_type,
    )

    validate_reducer_plan(plan)
    assert plan.output_length > 0


def test_timm_vit_qkv_prune_dim_compiles_head_dim_reducer_plan():
    model = make_tiny_vit()
    attention = model.blocks[0].attn
    group = qkv_dependency_group(model, attention)

    plan = compile_reducer_plan_from_groups(
        [group],
        operation_type=WeightOperationType.SUM,
        num_heads={attention.qkv: attention.num_heads},
        prune_dim=True,
    )

    validate_reducer_plan(plan)
    assert plan.output_length == attention.head_dim


def test_timm_vit_qkv_prune_num_heads_compiles_head_reducer_plan():
    model = make_tiny_vit()
    attention = model.blocks[0].attn
    group = qkv_dependency_group(model, attention)

    plan = compile_reducer_plan_from_groups(
        [group],
        operation_type=WeightOperationType.SUM,
        num_heads={attention.qkv: attention.num_heads},
        prune_num_heads=True,
    )

    validate_reducer_plan(plan)
    assert plan.output_length == attention.num_heads
