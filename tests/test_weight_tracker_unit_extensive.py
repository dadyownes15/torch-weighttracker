import pytest
import torch
import torch.nn as nn

from tests.test_calculation_specs import TinyLinearChain, _model_and_groups
from torch_weighttracker import WeightTracker
from torch_weighttracker.calculations import CalcType
from torch_weighttracker.consumer_ignore import (
    ModuleIgnore,
    without_ignored_canonical_members,
)
from torch_weighttracker.regularizers import RegularizerType
from torch_weighttracker.trackers import TrackerType


def _assert_tensor_dict_close(actual, expected: dict[str, torch.Tensor]) -> None:
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        torch.testing.assert_close(actual[key], expected_value)


def test_package_exports_weight_tracker() -> None:
    from torch_weighttracker.weight_tracker import WeightTracker as DirectTracker

    assert WeightTracker is DirectTracker


def test_dependency_build_requires_example_inputs_and_root_types_together() -> None:
    model = TinyLinearChain()

    with pytest.raises(ValueError, match="requires both example_inputs"):
        WeightTracker(model, root_module_types=[nn.Linear])


def test_weight_tracker_builds_groups_from_example_inputs() -> None:
    model = TinyLinearChain()
    tracker = WeightTracker(
        model,
        example_inputs=torch.randn(1, 2),
        root_module_types=[nn.Linear],
    )

    assert len(tracker.groups) == 2
    assert tuple(group.length for group in tracker.canonical_groups) == (1, 3)
    assert tracker.dependency_graph is not None


def test_view_structures_returns_canonical_group_printout() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker.__new__(WeightTracker)
    tracker.model = model
    tracker.canonical_groups = groups

    text = tracker.view_structures()

    assert "CanonicalGroups" in text
    assert "groups=2 total_units=4" in text
    assert "- group 0: units=[0:3) length=3 kind=channel members=2" in text
    assert (
        "    - fc1 Linear axis=out_channel layout=plain dest=(0, 1, 2)"
        in text
    )
    assert (
        "    - fc2 Linear axis=in_channel layout=plain dest=(0, 1, 2)"
        in text
    )
    assert "- group 1: units=[3:4) length=1 kind=channel members=1" in text
    assert "    - fc2 Linear axis=out_channel layout=plain dest=(3)" in text


def test_weight_tracker_removes_ignored_layer_members_from_groups() -> None:
    model = TinyLinearChain()
    tracker = WeightTracker(
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
    tracker = WeightTracker(model, groups=groups, device="cpu", dtype=torch.float64)

    units_to_group = tracker.get_calculation(CalcType.UNITS_TO_GROUP)
    baseline_sizes = tracker.get_calculation(CalcType.INIT_UNIT_PR_GROUP_COUNT)
    group_sizes = tracker.get_calculation(CalcType.GROUP_SIZES)

    unit_values = torch.ones(4, dtype=torch.float64)
    torch.testing.assert_close(
        units_to_group(unit_values),
        torch.tensor([3.0, 1.0], dtype=torch.float64),
    )
    assert units_to_group(unit_values).dtype == torch.float64
    assert baseline_sizes().dtype == torch.float64
    assert group_sizes().dtype == torch.long


def test_cached_constant_calculations_keep_baseline_after_weight_changes() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, groups=groups)

    baseline_sizes = tracker.get_calculation(CalcType.INIT_UNIT_PR_GROUP_COUNT)
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
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

    structured_bops = tracker.create_tracker(TrackerType.STRUCTURED_BOPS)
    metrics = structured_bops.track()

    assert metrics.keys() == {
        "structured_bops_compression",
        "structured_bops_compression_rate_pr_module",
    }
    torch.testing.assert_close(
        metrics["structured_bops_compression"],
        torch.tensor(1.0 - 96.0 / 9216.0),
    )
    _assert_tensor_dict_close(
        metrics["structured_bops_compression_rate_pr_module"],
        {
            "fc1": torch.tensor(1.0 - 64.0 / 6144.0),
            "fc2": torch.tensor(1.0 - 32.0 / 3072.0),
        },
    )
    assert tracker.trackers == [structured_bops]


def test_create_regularizer_wires_group_lasso_and_keeps_gradients() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, groups=groups)

    regularizer = tracker.create_regularizer(RegularizerType.GROUP_LASSO)
    loss = regularizer()
    loss.backward()

    assert loss.ndim == 0
    assert model.fc1.weight.grad is not None
    assert model.fc2.weight.grad is not None
    assert tracker.regularizers == [regularizer]


def test_group_lasso_ignore_is_invariant_to_ignored_module_weights() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, groups=groups)
    regularizer = tracker.create_regularizer(
        RegularizerType.GROUP_LASSO,
        ignore=[model.fc2],
    )

    before = regularizer()

    with torch.no_grad():
        model.fc2.weight.add_(1000.0)

    torch.testing.assert_close(regularizer(), before)

    with torch.no_grad():
        model.fc1.weight.add_(1.0)

    assert not torch.allclose(regularizer(), before)


def test_group_lasso_ignore_all_members_raises() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, groups=groups)

    with pytest.raises(ValueError, match="removed all canonical members"):
        tracker.create_regularizer(
            RegularizerType.GROUP_LASSO,
            ignore=[model],
        )


def test_structured_bops_ignore_all_weighted_modules_raises() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

    with pytest.raises(ValueError, match="removed all canonical members"):
        tracker.create_tracker(
            TrackerType.STRUCTURED_BOPS,
            ignore=[model],
        )


def test_structured_bops_ignore_filters_weighted_modules() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

    full = tracker.create_tracker(TrackerType.STRUCTURED_BOPS)
    filtered = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        ignore=[model.fc2],
    )

    assert full.compute().numel() == len(tracker._get_weighted_modules())
    assert filtered.compute().numel() == 1


def test_structured_bops_metric_module_names_follow_context() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

    full_metrics = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        log_module_names=True,
    ).track()
    filtered_metrics = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        ignore=[model.fc2],
        log_module_names=True,
    ).track()

    assert full_metrics["structured_bops_module_names"] == ("fc1", "fc2")
    assert filtered_metrics["structured_bops_module_names"] == ("fc1",)
    assert filtered_metrics["structured_bops_compression_rate_pr_module"].keys() == {
        "fc1"
    }


def test_structured_bops_default_compression_follows_context() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    with torch.no_grad():
        model.fc2.weight.zero_()
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

    full_metrics = tracker.create_tracker(TrackerType.STRUCTURED_BOPS).track()
    filtered_metrics = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        ignore=[model.fc2],
    ).track()

    assert full_metrics.keys() == {
        "structured_bops_compression",
        "structured_bops_compression_rate_pr_module",
    }
    torch.testing.assert_close(
        full_metrics["structured_bops_compression"],
        torch.tensor(1.0 - 64.0 / 9216.0),
    )
    _assert_tensor_dict_close(
        full_metrics["structured_bops_compression_rate_pr_module"],
        {
            "fc1": torch.tensor(1.0 - 64.0 / 6144.0),
            "fc2": torch.tensor(1.0),
        },
    )
    assert filtered_metrics.keys() == {
        "structured_bops_compression",
        "structured_bops_compression_rate_pr_module",
    }
    torch.testing.assert_close(
        filtered_metrics["structured_bops_compression"],
        torch.tensor(1.0 - 64.0 / 6144.0),
    )
    _assert_tensor_dict_close(
        filtered_metrics["structured_bops_compression_rate_pr_module"],
        {"fc1": torch.tensor(1.0 - 64.0 / 6144.0)},
    )


def test_structured_bops_total_bops_logging_is_opt_in() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

    metrics = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        log_total_bops=True,
        log_compression_rate=True,
    ).track()

    torch.testing.assert_close(metrics["structured_bops"], torch.tensor(96.0))
    torch.testing.assert_close(
        metrics["structured_bops_baseline"],
        torch.tensor(9216.0),
    )
    torch.testing.assert_close(
        metrics["structured_bops_compression_rate"],
        metrics["structured_bops_compression"],
    )
    _assert_tensor_dict_close(
        metrics["structured_bops_pr_module"],
        {"fc1": torch.tensor(64.0), "fc2": torch.tensor(32.0)},
    )
    _assert_tensor_dict_close(
        metrics["structured_bops_baseline_pr_module"],
        {"fc1": torch.tensor(6144.0), "fc2": torch.tensor(3072.0)},
    )


def test_group_lasso_ignore_uses_context_key() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, groups=groups)
    tracker.ensure_calculations((CalcType.L2_NORM_PR_UNIT,))
    global_l2 = tracker.calculations[CalcType.L2_NORM_PR_UNIT]

    regularizer = tracker.create_regularizer(
        RegularizerType.GROUP_LASSO,
        ignore=[model.fc2],
    )

    assert regularizer.calc(CalcType.L2_NORM_PR_UNIT) is not global_l2


def test_filtering_keeps_group_index_space_stable() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, groups=groups)

    filtered_groups = without_ignored_canonical_members(
        tracker.canonical_groups,
        ModuleIgnore([model.fc2]),
    )

    assert [
        (group.group_id, group.offset, group.length)
        for group in filtered_groups
    ] == [
        (group.group_id, group.offset, group.length)
        for group in tracker.canonical_groups
    ]
    assert all(
        member.module is not model.fc2
        for group in filtered_groups
        for member in group.members
    )


def test_consumer_ignore_does_not_mutate_global_calculations() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

    tracker.ensure_calculations((CalcType.BITRATE_PR_MODULE,))
    global_calc = tracker.calculations[CalcType.BITRATE_PR_MODULE]

    tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        ignore=[model.fc2],
    )

    assert tracker.calculations[CalcType.BITRATE_PR_MODULE] is global_calc


def test_same_ignore_reuses_context_keyed_calculation() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

    first = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        ignore=[model.fc2],
    )
    second = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        ignore=[model.fc2],
    )

    assert first.calc(CalcType.BITRATE_PR_MODULE) is second.calc(
        CalcType.BITRATE_PR_MODULE
    )


def test_consumer_kwargs_are_not_silently_dropped() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

    with pytest.raises(TypeError):
        tracker.create_tracker(
            TrackerType.STRUCTURED_BOPS,
            unsupported_option=True,
        )
