import timm
import pytest
import torch
import torch.nn as nn

from torch_structracker.calculations import CalculationType
from torch_structracker.operations import WeightOperationType
from torch_structracker.reducer_plan import (
    compile_reducer_plan_from_groups,
    validate_reducer_plan,
)
from torch_structracker.structure_tracker import StructureTracker
from torch_structracker.torch_pruning.dependency import DependencyGraph
from torch_structracker.trackers import TrackerType


class DirectMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(4, 2, batch_first=True)
        self.out = nn.Linear(4, 4)

    def forward(self, x):
        output, _ = self.mha(x, x, x, need_weights=False)
        return self.out(output)


class DirectMHABatchFirstFalse(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(4, 2, batch_first=False)

    def forward(self, x):
        output, _ = self.mha(x, x, x, need_weights=False)
        return output


class DirectMHAWithUnequalViews(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(6, 3, batch_first=True)
        self.out = nn.Linear(6, 6)

    def forward(self, x):
        output, _ = self.mha(x, x, x, need_weights=False)
        return self.out(output)


class SeparateProjectionMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            4,
            2,
            kdim=6,
            vdim=8,
            batch_first=True,
        )

    def forward(self, query, key, value):
        output, _ = self.mha(query, key, value, need_weights=False)
        return output


class TrackerInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class TupleInputTrackerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x, scale):
        return self.fc(x) * scale


class DictOutputTrackerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return {"logits": self.fc(x)}


class UnwrappedParameterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 3)
        self.scale = nn.Parameter(torch.ones(3))

    def forward(self, x):
        return self.fc(x) * self.scale


def make_tiny_vit():
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        img_size=32,
        num_classes=0,
    )
    model.eval()
    return model


def mha_dependency_groups(model, example_inputs=None):
    if example_inputs is None:
        example_inputs = torch.ones(2, 3, 4)

    graph = DependencyGraph().build_dependency(
        model=model,
        example_inputs=example_inputs,
    )
    return list(graph.get_all_groups(root_module_types=[nn.MultiheadAttention]))


@pytest.mark.parametrize("operation_type", list(WeightOperationType))
def test_direct_mha_dependency_groups_compile_reducer_plan_for_all_operations(
    operation_type,
):
    model = DirectMHA()
    groups = mha_dependency_groups(model)

    assert len(groups) == 1

    plan = compile_reducer_plan_from_groups(
        groups,
        operation_type=operation_type,
    )

    validate_reducer_plan(plan)
    assert plan.output_length == model.mha.embed_dim


def test_direct_mha_batch_first_false_dependency_groups_compile_reducer_plan():
    model = DirectMHABatchFirstFalse()
    groups = mha_dependency_groups(
        model,
        example_inputs=torch.ones(3, 2, 4),
    )

    assert len(groups) == 1

    plan = compile_reducer_plan_from_groups(
        groups,
        operation_type=WeightOperationType.SUM,
    )

    validate_reducer_plan(plan)
    assert plan.output_length == model.mha.embed_dim


def test_direct_mha_prune_dim_groups_compile_head_dim_reducer_plan():
    model = DirectMHAWithUnequalViews()
    groups = mha_dependency_groups(
        model,
        example_inputs=torch.ones(2, 3, 6),
    )

    plan = compile_reducer_plan_from_groups(
        groups,
        operation_type=WeightOperationType.SUM,
        num_heads={model.mha: model.mha.num_heads},
        prune_dim=True,
    )

    validate_reducer_plan(plan)
    assert plan.output_length == model.mha.head_dim


def test_direct_mha_prune_num_heads_groups_compile_head_reducer_plan():
    model = DirectMHAWithUnequalViews()
    groups = mha_dependency_groups(
        model,
        example_inputs=torch.ones(2, 3, 6),
    )

    plan = compile_reducer_plan_from_groups(
        groups,
        operation_type=WeightOperationType.SUM,
        num_heads={model.mha: model.mha.num_heads},
        prune_num_heads=True,
    )

    validate_reducer_plan(plan)
    assert plan.output_length == model.mha.num_heads


def test_direct_mha_separate_qkv_dependency_groups_compile_reducer_plan():
    model = SeparateProjectionMHA()
    groups = mha_dependency_groups(
        model,
        example_inputs=(
            torch.ones(2, 3, 4),
            torch.ones(2, 3, 6),
            torch.ones(2, 3, 8),
        ),
    )

    assert len(groups) == 1

    plan = compile_reducer_plan_from_groups(
        groups,
        operation_type=WeightOperationType.SUM,
    )

    validate_reducer_plan(plan)
    assert plan.output_length == model.mha.embed_dim


def test_structtracker_builds_dependency_groups_from_flat_graph_config():
    model = TrackerInputModel()
    tracker = StructureTracker(
        model,
        example_inputs=torch.ones(1, 2),
        root_module_types=[nn.Linear],
        ignored_layers=[],
    )

    tracker.create_tracker(TrackerType.PARAMETER_SUM)

    metrics = tracker.track()
    assert "parameter_sum" in metrics


def test_structtracker_accepts_tuple_example_inputs_for_dependency_graph():
    model = TupleInputTrackerModel()
    tracker = StructureTracker(
        model,
        example_inputs=(torch.ones(1, 2), torch.tensor(2.0)),
        root_module_types=[nn.Linear],
    )

    calculation = tracker.get_calculation(CalculationType.STRUCTURED_UNIT_SUM)
    assert calculation.output_length > 0


def test_structtracker_passes_forward_fn_and_output_transform_to_dependency_graph():
    model = DictOutputTrackerModel()

    def forward_fn(model, example_inputs):
        return model(example_inputs)

    tracker = StructureTracker(
        model,
        example_inputs=torch.ones(1, 2),
        forward_fn=forward_fn,
        output_transform=lambda output: output["logits"],
        root_module_types=[nn.Linear],
    )

    calculation = tracker.get_calculation(CalculationType.STRUCTURED_UNIT_SUM)
    assert calculation.output_length > 0


def test_structtracker_respects_ignored_layers_when_building_groups():
    model = TrackerInputModel()
    tracker = StructureTracker(
        model,
        example_inputs=torch.ones(1, 2),
        root_module_types=[nn.Linear],
        ignored_layers=[model.fc2],
    )

    assert len(tracker.groups) > 0
    assert all(
        member.dep.target.module is not model.fc2
        for group in tracker.groups
        for member in group.items
    )


def test_structtracker_accepts_ignored_params_when_building_dependency_graph():
    model = UnwrappedParameterModel()
    tracker = StructureTracker(
        model,
        example_inputs=torch.ones(1, 2),
        root_module_types=[nn.Linear],
        ignored_params=[model.scale],
    )

    calculation = tracker.get_calculation(CalculationType.STRUCTURED_UNIT_SUM)
    assert calculation.output_length > 0


def test_structtracker_registers_unwrapped_parameters_when_building_groups():
    model = UnwrappedParameterModel()
    tracker = StructureTracker(
        model,
        example_inputs=torch.ones(1, 2),
        root_module_types=[nn.Linear],
        unwrapped_parameters={model.scale: 0},
    )

    calculation = tracker.get_calculation(CalculationType.STRUCTURED_UNIT_SUM)
    assert calculation.output_length > 0


def test_structtracker_direct_mha_num_heads_config_creates_head_level_structure():
    model = DirectMHA()
    tracker = StructureTracker(
        model,
        example_inputs=torch.ones(2, 3, 4),
        root_module_types=[nn.MultiheadAttention],
        num_heads={model.mha: model.mha.num_heads},
        prune_num_heads=True,
    )

    calculation = tracker.get_calculation(CalculationType.STRUCTURED_UNIT_SUM)
    assert calculation.output_length == model.mha.num_heads


def test_structtracker_vit_num_heads_config_creates_head_level_qkv_structure():
    model = make_tiny_vit()
    first_attention = model.blocks[0].attn
    tracker = StructureTracker(
        model,
        example_inputs=torch.ones(1, 3, 32, 32),
        root_module_types=[nn.Linear],
        ignored_layers=[],
        num_heads={first_attention.qkv: first_attention.num_heads},
        prune_num_heads=True,
    )

    calculation = tracker.get_calculation(CalculationType.STRUCTURED_UNIT_SUM)

    assert calculation.output_length < 3 * first_attention.qkv.out_features
    assert first_attention.num_heads <= calculation.output_length


def test_structtracker_vit_without_num_heads_keeps_raw_qkv_channel_structure():
    model = make_tiny_vit()
    first_attention = model.blocks[0].attn
    tracker = StructureTracker(
        model,
        example_inputs=torch.ones(1, 3, 32, 32),
        root_module_types=[nn.Linear],
        ignored_layers=[],
    )

    calculation = tracker.get_calculation(CalculationType.STRUCTURED_UNIT_SUM)

    assert calculation.output_length >= first_attention.qkv.out_features
