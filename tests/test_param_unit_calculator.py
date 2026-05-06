import torch
import torch.nn as nn

from torch_structracker.regularizers import ParamUnitSum
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

    calculator = ParamUnitSum(groups)

    assert calculator.unit_count == 6
    torch.testing.assert_close(
        calculator(),
        torch.tensor([27.0, 37.0, 47.0, 20.0, 26.0, 32.0]),
    )
