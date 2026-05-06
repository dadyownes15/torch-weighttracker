import pytest
import torch
import torch.nn as nn

from torch_structracker.operations import WeightOperationType
from torch_structracker.reducer_plan import (
    compile_reducer_plan_from_groups,
    validate_reducer_plan,
)
from torch_structracker.torch_pruning.dependency import DependencyGraph


class TwoLayerLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=False)
        self.fc2 = nn.Linear(3, 1, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class ResidualConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4)
        self.shortcut = nn.Conv2d(3, 4, kernel_size=1, bias=False)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        return x + residual


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


@pytest.mark.parametrize(
    ("operation_type", "expected"),
    [
        (WeightOperationType.SUM, torch.tensor([24.0, 6.0, 7.0, 8.0])),
        (WeightOperationType.MEAN, torch.tensor([8.0, 6.5, 7.5, 8.5])),
        (WeightOperationType.COUNT, torch.tensor([3.0, 3.0, 3.0, 3.0])),
        (WeightOperationType.L1, torch.tensor([24.0, 10.0, 15.0, 20.0])),
        (
            WeightOperationType.L2,
            torch.tensor(
                [
                    13.928388595581055,
                    9.236067771911621,
                    13.0,
                    16.81024932861328,
                ]
            ),
        ),
    ],
)
def test_all_operations_execute_on_real_linear_dependency_groups(
    operation_type,
    expected,
):
    model = TwoLayerLinear()
    with torch.no_grad():
        model.fc1.weight.copy_(
            torch.tensor(
                [
                    [1.0, -2.0],
                    [3.0, -4.0],
                    [5.0, -6.0],
                ]
            )
        )
        model.fc2.weight.copy_(torch.tensor([[7.0, 8.0, 9.0]]))

    groups = dependency_groups(
        model=model,
        example_inputs=torch.ones(1, 2),
        root_module_types=[nn.Linear],
    )
    plan = compile_reducer_plan_from_groups(
        groups,
        operation_type=operation_type,
    )

    validate_reducer_plan(plan)
    assert plan.output_length == 4
    torch.testing.assert_close(accumulate_plan(plan), expected)


@pytest.mark.parametrize("operation_type", list(WeightOperationType))
def test_all_operations_validate_on_real_residual_conv_dependency_groups(
    operation_type,
):
    model = ResidualConvBlock().eval()

    groups = dependency_groups(
        model=model,
        example_inputs=torch.ones(1, 3, 2, 2),
        root_module_types=[nn.Conv2d],
    )
    plan = compile_reducer_plan_from_groups(
        groups,
        operation_type=operation_type,
    )

    validate_reducer_plan(plan)
    assert plan.output_length == 8
    assert len(plan.mappings) >= 5


def test_residual_dependency_graph_maps_multiple_members_to_same_output_units():
    model = ResidualConvBlock().eval()

    groups = dependency_groups(
        model=model,
        example_inputs=torch.ones(1, 3, 2, 2),
        root_module_types=[nn.Conv2d],
    )
    plan = compile_reducer_plan_from_groups(
        groups,
        operation_type=WeightOperationType.SUM,
    )

    destination_counts = {index: 0 for index in range(plan.output_length)}
    for mapping in plan.mappings:
        for destination_index in mapping.destination_indices:
            destination_counts[destination_index] += 1

    validate_reducer_plan(plan)
    assert plan.output_length == 8
    assert min(destination_counts.values()) > 1


def test_stale_reducer_plan_fails_after_structural_parameter_change():
    model = TwoLayerLinear()
    groups = dependency_groups(
        model=model,
        example_inputs=torch.ones(1, 2),
        root_module_types=[nn.Linear],
    )
    plan = compile_reducer_plan_from_groups(
        groups,
        operation_type=WeightOperationType.SUM,
    )

    validate_reducer_plan(plan)
    model.fc1.weight = nn.Parameter(model.fc1.weight[:2].detach().clone())

    with pytest.raises(ValueError, match="produced 2 values"):
        validate_reducer_plan(plan)
