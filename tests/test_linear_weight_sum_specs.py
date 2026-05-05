import torch
import torch.nn as nn

from torch_structracker.regularizers import GroupLasso
from torch_structracker.torch_pruning.dependency import DependencyGraph


class TwoLayerLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=False)
        self.fc2 = nn.Linear(3, 1, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def test_group_lasso_accumulates_linear_weight_sums_from_group_dependencies():
    model = TwoLayerLinear()
    with torch.no_grad():
        model.fc1.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ]
            )
        )
        model.fc2.weight.copy_(torch.tensor([[7.0, 8.0, 9.0]]))

    graph = DependencyGraph().build_dependency(
        model=model,
        example_inputs=torch.ones(1, 2),
    )
    groups = list(graph.get_all_groups(root_module_types=[nn.Linear]))

    regularizer = GroupLasso(groups)

    assert len(regularizer.specs) == 3
    assert torch.equal(
        regularizer(),
        torch.tensor([24.0, 10.0, 15.0, 20.0]),
    )
