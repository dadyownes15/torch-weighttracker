import torch
import torch.nn as nn

from torch_structracker import StructureTracker
from torch_structracker.calculations import CalculationType
from torch_structracker.torch_pruning.dependency import DependencyGraph
from torch_structracker.trackers import ParameterSumTracker, TrackerType


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


class GradProbeCalculation:
    def __call__(self):
        return torch.tensor([float(torch.is_grad_enabled())])


def linear_groups_from_tiny_mlp():
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
    return model, list(graph.get_all_groups(root_module_types=[nn.Linear]))


def test_parameter_sum_tracker_tracks_structured_unit_sum_metrics():
    model, groups = linear_groups_from_tiny_mlp()
    struct_tracker = StructureTracker(model=model, groups=groups)

    tracker = struct_tracker.create_tracker(TrackerType.PARAMETER_SUM)
    metrics = tracker.track()

    assert isinstance(tracker, ParameterSumTracker)
    torch.testing.assert_close(
        metrics["structured_unit_sum"],
        torch.tensor([27.0, 37.0, 47.0, 20.0, 26.0, 32.0]),
    )
    assert metrics["parameter_sum"] == 189.0


def test_parameter_sum_tracker_compute_is_no_grad():
    tracker = ParameterSumTracker(
        calculations={
            CalculationType.STRUCTURED_UNIT_SUM: GradProbeCalculation(),
        }
    )

    result = tracker.compute()

    torch.testing.assert_close(result, torch.tensor([0.0]))
