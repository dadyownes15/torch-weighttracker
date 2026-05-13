import pytest
import torch
import torch.nn as nn

from tests.test_calculation_specs import TinyLinearChain, _model_and_groups
from torch_structracker import StructureTracker
from torch_structracker.calculations import CalcType
from torch_structracker.regularizers import RegularizerType
from torch_structracker.trackers import TrackerType


def test_package_exports_structure_tracker() -> None:
    from torch_structracker.structure_tracker import StructureTracker as DirectTracker

    assert StructureTracker is DirectTracker


def test_dependency_build_requires_example_inputs_and_root_types_together() -> None:
    model = TinyLinearChain()

    with pytest.raises(ValueError, match="requires both example_inputs"):
        StructureTracker(model, example_inputs=torch.randn(1, 2))

    with pytest.raises(ValueError, match="requires both example_inputs"):
        StructureTracker(model, root_module_types=[nn.Linear])


def test_structure_tracker_builds_groups_from_example_inputs() -> None:
    model = TinyLinearChain()
    tracker = StructureTracker(
        model,
        example_inputs=torch.randn(1, 2),
        root_module_types=[nn.Linear],
    )

    assert len(tracker.groups) == 2
    assert tuple(group.length for group in tracker.canonical_groups) == (1, 3)
    assert tracker.dependency_graph is not None


def test_structure_tracker_removes_ignored_layer_members_from_groups() -> None:
    model = TinyLinearChain()
    tracker = StructureTracker(
        model,
        example_inputs=torch.randn(1, 2),
        root_module_types=[nn.Linear],
        ignored_layers=[model.fc2],
    )

    modules = {
        member.module
        for group in tracker.canonical_groups
        for member in group.members
    }
    assert model.fc1 in modules
    assert model.fc2 not in modules


def test_calculation_device_and_dtype_are_applied_to_pipeline_outputs() -> None:
    model, groups = _model_and_groups()
    tracker = StructureTracker(model, groups=groups, device="cpu", dtype=torch.float64)

    units_to_group = tracker.get_calculation(CalcType.UNITS_TO_GROUP)
    baseline_sizes = tracker.get_calculation(CalcType.BASELINE_GROUP_SIZES)
    group_sizes = tracker.get_calculation(CalcType.GROUP_SIZES)

    unit_values = torch.ones(4, dtype=torch.float64)
    torch.testing.assert_close(
        units_to_group(unit_values),
        torch.tensor([3.0, 1.0], dtype=torch.float64),
    )
    assert units_to_group(unit_values).dtype == torch.float64
    assert baseline_sizes().dtype == torch.float64
    assert group_sizes().dtype == torch.long


def test_cached_constant_calculations_keep_their_baseline_after_weight_changes() -> None:
    model, groups = _model_and_groups()
    tracker = StructureTracker(model, groups=groups)

    baseline_sizes = tracker.get_calculation(CalcType.BASELINE_GROUP_SIZES)
    active_units = tracker.get_calculation(CalcType.ACTIVE_UNITS)

    with torch.no_grad():
        model.fc1.weight[0, :] = 0
        model.fc2.weight[:, 0] = 0

    torch.testing.assert_close(baseline_sizes(), torch.tensor([3.0, 1.0]))
    torch.testing.assert_close(active_units(), torch.tensor([0.0, 0.0, 2.0, 1.0]))


def test_create_tracker_wires_structured_bops_from_required_calculations() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = StructureTracker(model, groups=groups)

    structured_bops = tracker.create_tracker(TrackerType.STRUCTURED_BOPS)
    metrics = structured_bops.track()

    torch.testing.assert_close(metrics["structured_bops"], torch.tensor(32.0))
    torch.testing.assert_close(
        metrics["structured_bops_pr_module"],
        torch.tensor([0.0, 32.0]),
    )
    assert tracker.trackers == [structured_bops]


def test_create_regularizer_wires_group_lasso_and_keeps_gradients() -> None:
    model, groups = _model_and_groups()
    tracker = StructureTracker(model, groups=groups)

    regularizer = tracker.create_regularizer(RegularizerType.GROUP_LASSO)
    loss = regularizer()
    loss.backward()

    assert loss.ndim == 0
    assert model.fc1.weight.grad is not None
    assert model.fc2.weight.grad is not None
    assert tracker.regularizers == [regularizer]
