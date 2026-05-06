import torch
import torch.nn as nn

from torch_structracker.operations import WeightOperationType
from torch_structracker.reducer_plan import (
    compile_reducer_plan_from_modules,
    validate_reducer_plan,
)


class TinyWeightedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, bias=False)
        self.fc = nn.Linear(3, 2, bias=False)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


def accumulate_plan(plan):
    output = torch.zeros(plan.output_length)

    for mapping in plan.mappings:
        destination_indices = torch.tensor(
            mapping.destination_indices,
            dtype=torch.long,
        )
        output.index_add_(0, destination_indices, mapping.reducer().reshape(-1))

    return output


def test_compile_reducer_plan_from_modules_sums_each_weighted_layer():
    model = TinyWeightedModel()
    with torch.no_grad():
        model.conv.weight.copy_(
            torch.tensor(
                [
                    [[[1.0, 2.0], [3.0, 4.0]]],
                    [[[5.0, 6.0], [7.0, 8.0]]],
                ]
            )
        )
        model.fc.weight.copy_(
            torch.tensor(
                [
                    [9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0],
                ]
            )
        )

    plan = compile_reducer_plan_from_modules(
        model,
        operation_type=WeightOperationType.SUM,
    )

    validate_reducer_plan(plan)

    assert plan.output_length == 2
    assert plan.output_labels == ("conv", "fc")
    torch.testing.assert_close(
        accumulate_plan(plan),
        torch.tensor([36.0, 69.0]),
    )
