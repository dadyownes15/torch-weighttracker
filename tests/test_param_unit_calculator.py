import torch
import torch.nn as nn

from torch_structracker.calculations import CalcType, MappedReductionCalculation
from torch_structracker.canonical_units import canonicalize_groups
from torch_structracker.operations import WeightOperationType
from torch_structracker.plans.unit_weight_operation_plan import create_group_member_plan
from torch_structracker.torch_pruning.dependency import DependencyGraph


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 3, bias=False),
            nn.Linear(3, 2, bias=False),
            nn.Linear(2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_param_unit_calculator_counts_linear_units_for_tiny_mlp():
    model = TinyMLP()
    with torch.no_grad():
        model.net[0].weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ]
            )
        )
        model.net[1].weight.copy_(
            torch.tensor(
                [
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ]
            )
        )
        model.net[2].weight.copy_(torch.tensor([[13.0, 14.0]]))

    graph = DependencyGraph().build_dependency(
        model=model,
        example_inputs=torch.ones(1, 2),
    )
    groups = list(graph.get_all_groups(root_module_types=[nn.Linear]))
    plan = create_group_member_plan(
        canonicalize_groups(groups),
        WeightOperationType.SUM,
    )

    calculator = MappedReductionCalculation(
        plan,
        calculation_type=CalcType.STRUCTURED_UNIT_SUM,
    )

    assert calculator.output_length == 6
    torch.testing.assert_close(
        calculator(),
        torch.tensor([27.0, 37.0, 47.0, 20.0, 26.0, 32.0]),
    )
