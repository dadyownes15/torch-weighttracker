import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from tests.test_calculation_specs import (
    TinyLinearChain,
    _model_and_groups,
    _tracker_from_groups,
)
from torch_weighttracker import WeightTracker
from torch_weighttracker.calculations import CalcType
from torch_weighttracker.consumer_ignore import (
    ConsumerFilter,
    filter_canonical_members,
)
from torch_weighttracker.regularizers import RegularizerType
from torch_weighttracker.trackers import TrackerType, tracker_class_for_type


class FakeQuantZeroSmall(nn.Module):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.where(weight.abs() < 0.5, torch.zeros_like(weight), weight)


class TinyAttentionSparsityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=4, num_heads=2, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        return y


class TinyConvBatchNormHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 6 * 6, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.fc(x.flatten(1))


def _assert_tensor_dict_close(actual, expected: dict[str, torch.Tensor]) -> None:
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        torch.testing.assert_close(actual[key], expected_value)


def test_package_exports_weight_tracker() -> None:
    from torch_weighttracker import FakePruneUnitResult, PruneUnitResult
    from torch_weighttracker.weight_tracker import (
        FakePruneUnitResult as DirectFakePruneUnitResult,
    )
    from torch_weighttracker.weight_tracker import (
        PruneUnitResult as DirectPruneUnitResult,
    )
    from torch_weighttracker.weight_tracker import (
        WeightTracker as DirectTracker,
    )

    assert WeightTracker is DirectTracker
    assert FakePruneUnitResult is DirectFakePruneUnitResult
    assert PruneUnitResult is DirectPruneUnitResult


def test_weight_tracker_without_example_inputs_has_no_dependency_groups() -> None:
    model = TinyLinearChain()

    tracker = WeightTracker(model, root_module_types=[nn.Linear])

    assert tracker.groups == []
    assert tracker.canonical_groups == ()
    assert tracker.dependency_graph is None


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


@pytest.mark.parametrize(
    "example_inputs",
    (
        torch.empty(1, 2, device="meta"),
        (torch.empty(1, 2, device="meta"),),
        {"x": torch.empty(1, 2, device="meta")},
    ),
)
def test_weight_tracker_rejects_example_inputs_on_different_device(
    example_inputs,
) -> None:
    model = TinyLinearChain()

    with pytest.raises(ValueError, match="same device as the model"):
        WeightTracker(model, example_inputs=example_inputs)


def test_view_structures_returns_canonical_group_printout() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker.__new__(WeightTracker)
    tracker.model = model
    tracker.canonical_groups = groups

    text = tracker.view_structures()

    assert "CanonicalGroups" in text
    assert "groups=2 total_units=4" in text
    assert "- group 0: units=[0:3) length=3 kind=channel members=2" in text
    assert "    - fc1 Linear axis=out_channel layout=plain dest=(0, 1, 2)" in text
    assert "    - fc2 Linear axis=in_channel layout=plain dest=(0, 1, 2)" in text
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
        member.module for group in tracker.canonical_groups for member in group.members
    }
    assert model.fc1 in modules
    assert model.fc2 not in modules


def test_calculation_device_and_dtype_are_applied_to_pipeline_outputs() -> None:
    model, groups = _model_and_groups()
    tracker = _tracker_from_groups(model, groups, device="cpu", dtype=torch.float64)

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
    tracker = _tracker_from_groups(model, groups)

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
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    structured_bops = tracker.create_tracker(TrackerType.STRUCTURED_BOPS)
    metrics = structured_bops.track()

    assert metrics.keys() == {
        "structured_bops_compression",
    }
    torch.testing.assert_close(
        metrics["structured_bops_compression"],
        torch.tensor(1.0 - 96.0 / 9216.0),
    )
    assert tracker.trackers == [structured_bops]


def test_create_tracker_wires_unstructured_bops_from_required_calculations() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    unstructured_bops = tracker.create_tracker(TrackerType.UNSTRUCTURED_BOPS)
    metrics = unstructured_bops.track()

    assert metrics.keys() == {
        "unstructured_bops_compression",
    }
    torch.testing.assert_close(
        metrics["unstructured_bops_compression"],
        torch.tensor(1.0 - 80.0 / 9216.0),
    )
    assert tracker.trackers == [unstructured_bops]


@pytest.mark.parametrize(
    "tracker_types",
    (
        ["l2_norm_distribution", "group_pruning_summary"],
        [TrackerType.L2_NORM_DISTRIBUTION, TrackerType.GROUP_PRUNING_SUMMARY],
    ),
)
def test_create_tracker_accepts_tracker_type_lists(tracker_types) -> None:
    model, groups = _model_and_groups()
    tracker = _tracker_from_groups(model, groups)

    created_trackers = tracker.create_tracker(tracker_types)
    metrics = tracker.track()

    assert tracker.trackers == created_trackers
    assert len(created_trackers) == 2
    assert "l2_norm_distribution/fc1:prune_out_channels" in metrics
    assert "group_pruning/pruned_units" in metrics


def test_invalid_tracker_type_lists_available_tracker_values() -> None:
    model, groups = _model_and_groups()
    tracker = _tracker_from_groups(model, groups)

    with pytest.raises(ValueError) as exc_info:
        tracker.create_tracker("l2_norm")

    message = str(exc_info.value)
    assert "'l2_norm' is not a valid TrackerType" in message
    assert "Available trackers:" in message
    for tracker_type in TrackerType:
        assert tracker_type.value in message


def test_tracker_class_for_type_lists_available_tracker_values() -> None:
    with pytest.raises(ValueError) as exc_info:
        tracker_class_for_type("l2_norm")

    message = str(exc_info.value)
    assert "'l2_norm' is not a valid TrackerType" in message
    assert "Available trackers:" in message
    for tracker_type in TrackerType:
        assert tracker_type.value in message


def test_create_regularizer_wires_group_lasso_and_keeps_gradients() -> None:
    model, groups = _model_and_groups()
    tracker = _tracker_from_groups(model, groups)

    regularizer = tracker.create_regularizer(RegularizerType.GROUP_LASSO)
    loss = regularizer()
    loss.backward()

    assert loss.ndim == 0
    assert model.fc1.weight.grad is not None
    assert model.fc2.weight.grad is not None
    assert tracker.regularizers == [regularizer]


def test_group_pruning_summary_reports_flat_unit_and_param_counts() -> None:
    model, groups = _model_and_groups()
    tracker = _tracker_from_groups(model, groups)

    metrics = tracker.create_tracker(TrackerType.GROUP_PRUNING_SUMMARY).track()

    assert set(metrics) == {
        "group_pruning/pruned_units",
        "group_pruning/pruned_params",
        "group_pruning/groups/fc1:prune_out_channels/pruned_units",
        "group_pruning/groups/fc1:prune_out_channels/pruned_params",
        "group_pruning/groups/fc2:prune_out_channels/pruned_units",
        "group_pruning/groups/fc2:prune_out_channels/pruned_params",
    }
    assert all(not isinstance(value, dict) for value in metrics.values())
    torch.testing.assert_close(
        metrics["group_pruning/pruned_units"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        metrics["group_pruning/pruned_params"],
        torch.tensor(4.0),
    )
    torch.testing.assert_close(
        metrics["group_pruning/groups/fc1:prune_out_channels/pruned_units"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        metrics["group_pruning/groups/fc1:prune_out_channels/pruned_params"],
        torch.tensor(3.0),
    )
    torch.testing.assert_close(
        metrics["group_pruning/groups/fc2:prune_out_channels/pruned_units"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        metrics["group_pruning/groups/fc2:prune_out_channels/pruned_params"],
        torch.tensor(1.0),
    )


def test_group_pruning_summary_filters_canonical_members() -> None:
    model, groups = _model_and_groups()
    tracker = _tracker_from_groups(model, groups)

    metrics = tracker.create_tracker(
        TrackerType.GROUP_PRUNING_SUMMARY,
        ignore=[model.fc2],
    ).track()

    torch.testing.assert_close(
        metrics["group_pruning/pruned_units"],
        torch.tensor(2.0),
    )
    torch.testing.assert_close(
        metrics["group_pruning/pruned_params"],
        torch.tensor(2.0),
    )
    torch.testing.assert_close(
        metrics["group_pruning/groups/fc1:prune_out_channels/pruned_units"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        metrics["group_pruning/groups/fc1:prune_out_channels/pruned_params"],
        torch.tensor(2.0),
    )
    torch.testing.assert_close(
        metrics["group_pruning/groups/fc2:prune_out_channels/pruned_units"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        metrics["group_pruning/groups/fc2:prune_out_channels/pruned_params"],
        torch.tensor(0.0),
    )


def test_group_lasso_ignore_is_invariant_to_ignored_module_weights() -> None:
    model, groups = _model_and_groups()
    tracker = _tracker_from_groups(model, groups)
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
    tracker = _tracker_from_groups(model, groups)

    with pytest.raises(ValueError, match="removed all canonical members"):
        tracker.create_regularizer(
            RegularizerType.GROUP_LASSO,
            ignore=[model],
        )


def test_structured_bops_ignore_all_weighted_modules_raises() -> None:
    model, groups = _model_and_groups()
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    with pytest.raises(ValueError, match="removed all canonical members"):
        tracker.create_tracker(
            TrackerType.STRUCTURED_BOPS,
            ignore=[model],
        )


def test_unstructured_bops_filtering_all_weighted_modules_raises() -> None:
    model, groups = _model_and_groups()
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    with pytest.raises(ValueError, match="weighted modules"):
        tracker.create_tracker(
            TrackerType.UNSTRUCTURED_BOPS,
            ignore=[model],
        )


def test_structured_bops_ignore_filters_weighted_modules() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    full = tracker.create_tracker(TrackerType.STRUCTURED_BOPS)
    filtered = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        ignore=[model.fc2],
    )

    assert full.compute().numel() == len(tracker._get_weighted_modules())
    assert filtered.compute().numel() == 1


def test_structured_bops_include_filters_weighted_modules_and_names() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    metrics = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        include=[model.fc1],
        log_module_names=True,
        log_total_bops=True,
        log_layerwise_stats=True,
    ).track()

    assert metrics["structured_bops_module_names"] == ("fc1",)
    _assert_tensor_dict_close(
        metrics["structured_bops_pr_module"],
        {"fc1": torch.tensor(64.0)},
    )
    _assert_tensor_dict_close(
        metrics["structured_bops_baseline_pr_module"],
        {"fc1": torch.tensor(6144.0)},
    )
    assert metrics["structured_bops_compression_rate_pr_module"].keys() == {"fc1"}


def test_structured_bops_include_parent_matches_default_values() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    full_metrics = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        log_module_names=True,
        log_total_bops=True,
        log_layerwise_stats=True,
        log_compression_rate=True,
    ).track()
    included_metrics = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        include=[model],
        log_module_names=True,
        log_total_bops=True,
        log_layerwise_stats=True,
        log_compression_rate=True,
    ).track()

    assert (
        included_metrics["structured_bops_module_names"]
        == full_metrics["structured_bops_module_names"]
    )
    torch.testing.assert_close(
        included_metrics["structured_bops"],
        full_metrics["structured_bops"],
    )
    torch.testing.assert_close(
        included_metrics["structured_bops_baseline"],
        full_metrics["structured_bops_baseline"],
    )
    torch.testing.assert_close(
        included_metrics["structured_bops_compression"],
        full_metrics["structured_bops_compression"],
    )
    torch.testing.assert_close(
        included_metrics["structured_bops_compression_rate"],
        full_metrics["structured_bops_compression_rate"],
    )
    _assert_tensor_dict_close(
        included_metrics["structured_bops_pr_module"],
        full_metrics["structured_bops_pr_module"],
    )
    _assert_tensor_dict_close(
        included_metrics["structured_bops_baseline_pr_module"],
        full_metrics["structured_bops_baseline_pr_module"],
    )
    _assert_tensor_dict_close(
        included_metrics["structured_bops_compression_rate_pr_module"],
        full_metrics["structured_bops_compression_rate_pr_module"],
    )


def test_structured_bops_include_and_ignore_same_module_raises() -> None:
    model, groups = _model_and_groups()
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    with pytest.raises(
        ValueError,
        match="Consumer filters removed all canonical members",
    ):
        tracker.create_tracker(
            TrackerType.STRUCTURED_BOPS,
            include=[model.fc1],
            ignore=[model.fc1],
        )


def test_structured_bops_include_parent_ignore_child_keeps_fc1() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    metrics = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        include=[model],
        ignore=[model.fc2],
        log_module_names=True,
        log_total_bops=True,
        log_layerwise_stats=True,
    ).track()

    assert metrics["structured_bops_module_names"] == ("fc1",)
    _assert_tensor_dict_close(
        metrics["structured_bops_pr_module"],
        {"fc1": torch.tensor(64.0)},
    )


def test_bops_trackers_default_ignore_normalization_weighted_modules() -> None:
    model = TinyConvBatchNormHead().eval()
    tracker = WeightTracker(
        model,
        example_inputs=torch.randn(1, 3, 8, 8),
        root_module_types=[nn.Conv2d, nn.Linear],
    )

    assert tuple(name for name, _ in tracker._get_weighted_module_entries()) == (
        "conv",
        "bn",
        "fc",
    )

    structured_metrics = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        log_module_names=True,
        log_total_bops=True,
        log_layerwise_stats=True,
    ).track()
    unstructured_metrics = tracker.create_tracker(
        TrackerType.UNSTRUCTURED_BOPS,
        log_module_names=True,
        log_total_bops=True,
        log_layerwise_stats=True,
    ).track()

    assert structured_metrics["structured_bops_module_names"] == ("conv", "fc")
    assert unstructured_metrics["unstructured_bops_module_names"] == ("conv", "fc")
    assert tuple(structured_metrics["structured_bops_pr_module"]) == ("conv", "fc")
    assert tuple(unstructured_metrics["unstructured_bops_pr_module"]) == (
        "conv",
        "fc",
    )


def test_unstructured_sparsity_reports_weighted_total_and_layers() -> None:
    model, _ = _model_and_groups()
    tracker = WeightTracker(model)

    metrics = tracker.create_tracker(TrackerType.UNSTRUCTURED_SPARSITY).track()

    torch.testing.assert_close(
        metrics["unstructured_sparsity"],
        torch.tensor(4.0 / 9.0),
    )
    _assert_tensor_dict_close(
        metrics["layers"],
        {
            "fc1": torch.tensor(3.0 / 6.0),
            "fc2": torch.tensor(1.0 / 3.0),
        },
    )


def test_unstructured_sparsity_reflects_weight_changes_after_creation() -> None:
    model, _ = _model_and_groups()
    tracker = WeightTracker(model)
    sparsity = tracker.create_tracker(TrackerType.UNSTRUCTURED_SPARSITY)

    before = sparsity.track()

    with torch.no_grad():
        model.fc2.weight.zero_()

    after = sparsity.track()

    torch.testing.assert_close(before["unstructured_sparsity"], torch.tensor(4.0 / 9.0))
    torch.testing.assert_close(after["unstructured_sparsity"], torch.tensor(6.0 / 9.0))
    _assert_tensor_dict_close(
        after["layers"],
        {
            "fc1": torch.tensor(3.0 / 6.0),
            "fc2": torch.tensor(1.0),
        },
    )


def test_unstructured_sparsity_include_and_ignore_filter_layers() -> None:
    model, _ = _model_and_groups()
    tracker = WeightTracker(model)

    included = tracker.create_tracker(
        TrackerType.UNSTRUCTURED_SPARSITY,
        include=[model.fc1],
    ).track()
    ignored = tracker.create_tracker(
        TrackerType.UNSTRUCTURED_SPARSITY,
        ignore=[model.fc2],
    ).track()
    parent_minus_child = tracker.create_tracker(
        TrackerType.UNSTRUCTURED_SPARSITY,
        include=[model],
        ignore=[model.fc2],
    ).track()

    for metrics in (included, ignored, parent_minus_child):
        torch.testing.assert_close(metrics["unstructured_sparsity"], torch.tensor(0.5))
        _assert_tensor_dict_close(metrics["layers"], {"fc1": torch.tensor(0.5)})


def test_unstructured_sparsity_filtering_all_weighted_modules_raises() -> None:
    model, _ = _model_and_groups()
    tracker = WeightTracker(model)

    with pytest.raises(ValueError, match="weighted modules"):
        tracker.create_tracker(
            TrackerType.UNSTRUCTURED_SPARSITY,
            ignore=[model],
        )


def test_unstructured_sparsity_reads_effective_parametrized_weight() -> None:
    model = nn.Sequential(nn.Linear(4, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0.1, -0.2, 1.0, -1.0]]))
    parametrize.register_parametrization(model[0], "weight", FakeQuantZeroSmall())

    metrics = (
        WeightTracker(model)
        .create_tracker(
            TrackerType.UNSTRUCTURED_SPARSITY,
        )
        .track()
    )

    torch.testing.assert_close(metrics["unstructured_sparsity"], torch.tensor(0.5))
    _assert_tensor_dict_close(metrics["layers"], {"0": torch.tensor(0.5)})


def test_unstructured_sparsity_counts_multihead_attention_projection_weights() -> None:
    model = TinyAttentionSparsityModel()
    with torch.no_grad():
        model.attn.in_proj_weight[:6].zero_()
    tracker = WeightTracker(model)

    metrics = tracker.create_tracker(
        TrackerType.UNSTRUCTURED_SPARSITY,
        include=[model.attn],
        ignore=[model.attn.out_proj],
    ).track()

    torch.testing.assert_close(metrics["unstructured_sparsity"], torch.tensor(0.5))
    _assert_tensor_dict_close(metrics["layers"], {"attn": torch.tensor(0.5)})


def test_structured_bops_metric_module_names_follow_context() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    full_metrics = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        log_module_names=True,
    ).track()
    filtered_metrics = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        ignore=[model.fc2],
        log_module_names=True,
        log_layerwise_stats=True,
    ).track()

    assert full_metrics["structured_bops_module_names"] == ("fc1", "fc2")
    assert filtered_metrics["structured_bops_module_names"] == ("fc1",)
    assert filtered_metrics["structured_bops_compression_rate_pr_module"].keys() == {
        "fc1"
    }


def test_unstructured_bops_metric_module_names_follow_context() -> None:
    model, groups = _model_and_groups()
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    full_metrics = tracker.create_tracker(
        "unstructured_bops",
        log_module_names=True,
    ).track()
    filtered_metrics = tracker.create_tracker(
        TrackerType.UNSTRUCTURED_BOPS,
        ignore=[model.fc2],
        log_module_names=True,
        log_layerwise_stats=True,
    ).track()

    assert full_metrics["unstructured_bops_module_names"] == ("fc1", "fc2")
    assert filtered_metrics["unstructured_bops_module_names"] == ("fc1",)
    assert filtered_metrics["unstructured_bops_compression_rate_pr_module"].keys() == {
        "fc1"
    }


def test_structured_bops_default_compression_follows_context() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    with torch.no_grad():
        model.fc2.weight.zero_()
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    full_metrics = tracker.create_tracker(TrackerType.STRUCTURED_BOPS).track()
    filtered_metrics = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        ignore=[model.fc2],
    ).track()

    assert full_metrics.keys() == {
        "structured_bops_compression",
    }
    torch.testing.assert_close(
        full_metrics["structured_bops_compression"],
        torch.tensor(1.0 - 64.0 / 9216.0),
    )
    assert filtered_metrics.keys() == {
        "structured_bops_compression",
    }
    torch.testing.assert_close(
        filtered_metrics["structured_bops_compression"],
        torch.tensor(1.0 - 64.0 / 6144.0),
    )


def test_unstructured_bops_default_compression_follows_context() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    full_metrics = tracker.create_tracker(TrackerType.UNSTRUCTURED_BOPS).track()
    filtered_metrics = tracker.create_tracker(
        TrackerType.UNSTRUCTURED_BOPS,
        ignore=[model.fc2],
    ).track()

    assert full_metrics.keys() == {
        "unstructured_bops_compression",
    }
    torch.testing.assert_close(
        full_metrics["unstructured_bops_compression"],
        torch.tensor(1.0 - 80.0 / 9216.0),
    )
    assert filtered_metrics.keys() == {
        "unstructured_bops_compression",
    }
    torch.testing.assert_close(
        filtered_metrics["unstructured_bops_compression"],
        torch.tensor(1.0 - 48.0 / 6144.0),
    )


def test_structured_bops_layerwise_stats_are_opt_in() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    total_only = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        log_total_bops=True,
    ).track()
    layerwise = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        log_layerwise_stats=True,
    ).track()

    assert total_only.keys() == {
        "structured_bops_compression",
        "structured_bops",
        "structured_bops_baseline",
    }
    assert layerwise.keys() == {
        "structured_bops_compression",
        "structured_bops_compression_rate_pr_module",
    }
    _assert_tensor_dict_close(
        layerwise["structured_bops_compression_rate_pr_module"],
        {
            "fc1": torch.tensor(1.0 - 64.0 / 6144.0),
            "fc2": torch.tensor(1.0 - 32.0 / 3072.0),
        },
    )


def test_unstructured_bops_layerwise_stats_are_opt_in() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    total_only = tracker.create_tracker(
        TrackerType.UNSTRUCTURED_BOPS,
        log_total_bops=True,
    ).track()
    layerwise = tracker.create_tracker(
        TrackerType.UNSTRUCTURED_BOPS,
        log_layerwise_stats=True,
    ).track()

    assert total_only.keys() == {
        "unstructured_bops_compression",
        "unstructured_bops",
        "unstructured_bops_baseline",
    }
    assert layerwise.keys() == {
        "unstructured_bops_compression",
        "unstructured_bops_compression_rate_pr_module",
    }
    _assert_tensor_dict_close(
        layerwise["unstructured_bops_compression_rate_pr_module"],
        {
            "fc1": torch.tensor(1.0 - 48.0 / 6144.0),
            "fc2": torch.tensor(1.0 - 32.0 / 3072.0),
        },
    )


def test_structured_bops_total_bops_logging_is_opt_in() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    metrics = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        log_total_bops=True,
        log_layerwise_stats=True,
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


def test_unstructured_bops_total_bops_logging_is_opt_in() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    metrics = tracker.create_tracker(
        TrackerType.UNSTRUCTURED_BOPS,
        log_total_bops=True,
        log_layerwise_stats=True,
        log_compression_rate=True,
    ).track()

    torch.testing.assert_close(metrics["unstructured_bops"], torch.tensor(80.0))
    torch.testing.assert_close(
        metrics["unstructured_bops_baseline"],
        torch.tensor(9216.0),
    )
    torch.testing.assert_close(
        metrics["unstructured_bops_compression_rate"],
        metrics["unstructured_bops_compression"],
    )
    _assert_tensor_dict_close(
        metrics["unstructured_bops_pr_module"],
        {"fc1": torch.tensor(48.0), "fc2": torch.tensor(32.0)},
    )
    _assert_tensor_dict_close(
        metrics["unstructured_bops_baseline_pr_module"],
        {"fc1": torch.tensor(6144.0), "fc2": torch.tensor(3072.0)},
    )


def test_unstructured_bops_reflects_weight_changes_after_creation() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))
    unstructured_bops = tracker.create_tracker(
        TrackerType.UNSTRUCTURED_BOPS,
        log_total_bops=True,
        log_layerwise_stats=True,
    )

    before = unstructured_bops.track()

    with torch.no_grad():
        model.fc2.weight.zero_()

    after = unstructured_bops.track()

    torch.testing.assert_close(before["unstructured_bops"], torch.tensor(80.0))
    torch.testing.assert_close(after["unstructured_bops"], torch.tensor(48.0))
    torch.testing.assert_close(
        after["unstructured_bops_baseline"],
        before["unstructured_bops_baseline"],
    )
    _assert_tensor_dict_close(
        after["unstructured_bops_pr_module"],
        {"fc1": torch.tensor(48.0), "fc2": torch.tensor(0.0)},
    )


def test_group_lasso_ignore_uses_context_key() -> None:
    model, groups = _model_and_groups()
    tracker = _tracker_from_groups(model, groups)
    tracker.ensure_calculations((CalcType.L2_NORM_PR_UNIT,))
    global_l2 = tracker.calculations[CalcType.L2_NORM_PR_UNIT]

    regularizer = tracker.create_regularizer(
        RegularizerType.GROUP_LASSO,
        ignore=[model.fc2],
    )

    assert regularizer.calc(CalcType.L2_NORM_PR_UNIT) is not global_l2


def test_filtering_keeps_group_index_space_stable() -> None:
    model, groups = _model_and_groups()
    tracker = _tracker_from_groups(model, groups)

    filtered_groups = filter_canonical_members(
        tracker.canonical_groups,
        ConsumerFilter(ignore=[model.fc2]),
    )

    assert [
        (group.group_id, group.offset, group.length) for group in filtered_groups
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
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    tracker.ensure_calculations((CalcType.BITRATE_PR_MODULE,))
    global_calc = tracker.calculations[CalcType.BITRATE_PR_MODULE]

    tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        ignore=[model.fc2],
    )

    assert tracker.calculations[CalcType.BITRATE_PR_MODULE] is global_calc


def test_same_ignore_reuses_context_keyed_calculation() -> None:
    model, groups = _model_and_groups()
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

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
    tracker = _tracker_from_groups(model, groups, example_inputs=torch.randn(1, 2))

    with pytest.raises(TypeError):
        tracker.create_tracker(
            TrackerType.STRUCTURED_BOPS,
            unsupported_option=True,
        )


def _dependency_built_linear_tracker() -> WeightTracker:
    model = TinyLinearChain()
    with torch.no_grad():
        model.fc1.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 0.0],
                    [2.0, 3.0],
                ]
            )
        )
        model.fc2.weight.copy_(torch.tensor([[4.0, 0.0, 6.0]]))

    return WeightTracker(
        model,
        example_inputs=torch.randn(1, 2),
        root_module_types=[nn.Linear],
    )


class TinyBiasedLinearChain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class TinyConvBnNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


def _dependency_built_biased_linear_tracker() -> WeightTracker:
    model = TinyBiasedLinearChain()
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
        model.fc1.bias.copy_(torch.tensor([7.0, 8.0, 9.0]))
        model.fc2.weight.copy_(torch.tensor([[10.0, 11.0, 12.0]]))
        model.fc2.bias.copy_(torch.tensor([13.0]))

    return WeightTracker(
        model,
        example_inputs=torch.randn(1, 2),
        root_module_types=[nn.Linear],
    )


def _hidden_linear_group_id(tracker: WeightTracker) -> int:
    for group_id, group in enumerate(tracker.canonical_groups):
        modules = {member.module for member in group.members}
        if len(group.members) == 2 and group.length == 3:
            return group_id
    raise AssertionError(f"hidden linear group not found: {modules}")


def _output_linear_group_id(tracker: WeightTracker) -> int:
    for group_id, group in enumerate(tracker.canonical_groups):
        modules = {member.module for member in group.members}
        if len(group.members) == 1 and group.length == 1:
            return group_id
    raise AssertionError(f"output linear group not found: {modules}")


def test_fake_prune_unit_zeroes_output_member_bias_when_requested() -> None:
    tracker = _dependency_built_biased_linear_tracker()
    model = tracker.model
    group_id = _output_linear_group_id(tracker)

    result = tracker.fake_prune_unit(group_id, 0, prune_bias=True)

    assert result.group_id == group_id
    assert result.unit_id == 0
    assert result.zeroed_members == 1
    assert result.prune_bias is True
    assert result.zeroed_parameters == ("fc2.weight", "fc2.bias")
    torch.testing.assert_close(model.fc2.weight, torch.zeros_like(model.fc2.weight))
    torch.testing.assert_close(model.fc2.bias, torch.zeros_like(model.fc2.bias))
    assert model.fc2.weight.shape == (1, 3)


def test_fake_prune_unit_can_leave_output_bias_unchanged() -> None:
    tracker = _dependency_built_biased_linear_tracker()
    model = tracker.model
    group_id = _output_linear_group_id(tracker)

    result = tracker.fake_prune_unit(group_id, 0, prune_bias=False)

    assert result.zeroed_parameters == ("fc2.weight",)
    assert result.prune_bias is False
    torch.testing.assert_close(model.fc2.weight, torch.zeros_like(model.fc2.weight))
    torch.testing.assert_close(model.fc2.bias, torch.tensor([13.0]))


def test_fake_prune_unit_zeroes_hidden_output_row_and_input_column() -> None:
    tracker = _dependency_built_biased_linear_tracker()
    model = tracker.model
    group_id = _hidden_linear_group_id(tracker)

    result = tracker.fake_prune_unit(group_id, 1, prune_bias=True)

    assert result.zeroed_members == 2
    assert result.zeroed_parameters == ("fc1.weight", "fc1.bias", "fc2.weight")
    torch.testing.assert_close(model.fc1.weight[1], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(model.fc1.bias, torch.tensor([7.0, 0.0, 9.0]))
    torch.testing.assert_close(model.fc2.weight[:, 1], torch.tensor([0.0]))
    torch.testing.assert_close(model.fc2.bias, torch.tensor([13.0]))


def test_fake_prune_unit_keeps_consumers_and_zero_view_recomputes() -> None:
    tracker = _dependency_built_biased_linear_tracker()
    group_id = _hidden_linear_group_id(tracker)
    l2_tracker = tracker.create_tracker(TrackerType.L2_NORM_DISTRIBUTION)
    regularizer = tracker.create_regularizer(RegularizerType.GROUP_LASSO)
    l2_norm_calc = tracker.ensure_calculations((CalcType.L2_NORM_PR_UNIT,))[
        CalcType.L2_NORM_PR_UNIT
    ]
    tracker._get_weighted_modules()
    calculations_before = dict(tracker.calculations)
    weighted_module_entries = tracker._weighted_module_entries
    weighted_modules = tracker._weighted_modules
    weighted_module_index = tracker._weighted_module_index

    result = tracker.fake_prune_unit(group_id, 2)

    assert result.zeroed_members == 2
    assert tracker.trackers == [l2_tracker]
    assert tracker.regularizers == [regularizer]
    assert tracker.calculations == calculations_before
    assert tracker._weighted_module_entries is weighted_module_entries
    assert tracker._weighted_modules is weighted_modules
    assert tracker._weighted_module_index is weighted_module_index
    canonical_id = tracker.canonical_groups[group_id].offset + 2
    assert l2_norm_calc()[canonical_id] == 0

    view = tracker.view_zero_units()
    assert view.total_zero_units == 1
    assert view.groups[0].group_id == group_id
    assert view.groups[0].zero_units[0].unit_id == 2


def test_fake_prune_unit_invalid_unit_does_not_mutate_or_invalidate() -> None:
    tracker = _dependency_built_biased_linear_tracker()
    model = tracker.model
    group_id = _hidden_linear_group_id(tracker)
    before_fc1 = model.fc1.weight.detach().clone()
    before_fc2 = model.fc2.weight.detach().clone()
    tracker.ensure_calculations((CalcType.L2_NORM_PR_UNIT,))
    calculations_before = dict(tracker.calculations)

    with pytest.raises(IndexError):
        tracker.fake_prune_unit(group_id, 99)

    torch.testing.assert_close(model.fc1.weight, before_fc1)
    torch.testing.assert_close(model.fc2.weight, before_fc2)
    assert tracker.calculations == calculations_before


def test_prune_unit_uses_get_prune_unit_mapping_and_refreshes_state() -> None:
    tracker = _dependency_built_linear_tracker()
    model = tracker.model
    group_id = _hidden_linear_group_id(tracker)
    _, expected_idxs = tracker.get_prune_unit(group_id, 1)

    tracker.create_tracker(TrackerType.L2_NORM_DISTRIBUTION)
    tracker.create_regularizer(RegularizerType.GROUP_LASSO)
    tracker.ensure_calculations((CalcType.L2_NORM_PR_UNIT,))
    assert tracker.trackers
    assert tracker.regularizers
    assert tracker.calculations

    result = tracker.prune_unit(group_id, 1)

    assert result.group_id == group_id
    assert result.unit_id == 1
    assert result.pruning_idxs == expected_idxs
    assert model.fc1.weight.shape == (2, 2)
    assert model.fc2.weight.shape == (1, 2)
    assert tracker.trackers == []
    assert tracker.regularizers == []
    assert tracker.calculations == {}
    assert tuple(group.length for group in tracker.canonical_groups) == (1, 2)
    assert model(torch.randn(1, 2)).shape == (1, 1)


def test_prune_zero_units_dry_run_keeps_consumers_and_model_state() -> None:
    tracker = _dependency_built_linear_tracker()
    model = tracker.model
    tracker.create_tracker(TrackerType.L2_NORM_DISTRIBUTION)
    tracker.create_regularizer(RegularizerType.GROUP_LASSO)
    tracker.ensure_calculations((CalcType.L2_NORM_PR_UNIT,))
    calculations_before = dict(tracker.calculations)

    result = tracker.prune_zero_units(dry_run=True)

    assert result.dry_run is True
    assert result.pruned_units == 0
    assert result.view.total_zero_units == 1
    assert model.fc1.weight.shape == (3, 2)
    assert model.fc2.weight.shape == (1, 3)
    assert tracker.trackers
    assert tracker.regularizers
    assert calculations_before.keys() <= tracker.calculations.keys()


def test_prune_zero_units_real_prune_invalidates_and_rebuilds() -> None:
    tracker = _dependency_built_linear_tracker()
    model = tracker.model
    tracker.create_tracker(TrackerType.L2_NORM_DISTRIBUTION)
    tracker.create_regularizer(RegularizerType.GROUP_LASSO)
    old_group_lengths = tuple(group.length for group in tracker.canonical_groups)

    result = tracker.prune_zero_units()

    assert result.dry_run is False
    assert result.pruned_units == 1
    assert result.view.total_zero_units == 1
    assert old_group_lengths == (1, 3)
    assert tuple(group.length for group in tracker.canonical_groups) == (1, 2)
    assert model.fc1.weight.shape == (2, 2)
    assert model.fc2.weight.shape == (1, 2)
    assert tracker.trackers == []
    assert tracker.regularizers == []
    assert tracker.calculations == {}


def test_view_zero_structures_ignore_module_type_filters_detection_only() -> None:
    model = TinyConvBnNet()
    with torch.no_grad():
        model.conv.weight[1].zero_()
        model.bn.weight.fill_(1.0)

    tracker = WeightTracker(
        model,
        example_inputs=torch.randn(1, 3, 4, 4),
        root_module_types=[nn.Conv2d],
    )

    unfiltered = tracker.view_zero_structures()
    filtered = tracker.view_zero_structures(ignore=[nn.BatchNorm2d])

    assert unfiltered.total_zero_units == 0
    assert filtered.total_zero_units == 1
    assert filtered.groups[0].zero_units[0].unit_id == 1


def test_prune_zero_structures_ignore_still_prunes_coupled_modules() -> None:
    model = TinyConvBnNet()
    with torch.no_grad():
        model.conv.weight[1].zero_()
        model.bn.weight.fill_(1.0)

    tracker = WeightTracker(
        model,
        example_inputs=torch.randn(1, 3, 4, 4),
        root_module_types=[nn.Conv2d],
    )

    result = tracker.prune_zero_structures(ignore=[nn.BatchNorm2d])

    assert result.pruned_units == 1
    assert result.view.total_zero_units == 1
    assert model.conv.out_channels == 3
    assert model.bn.num_features == 3


def test_view_structures_ignore_module_type_hides_members() -> None:
    model = TinyConvBnNet()
    tracker = WeightTracker(
        model,
        example_inputs=torch.randn(1, 3, 4, 4),
        root_module_types=[nn.Conv2d],
    )

    text = tracker.view_structures(ignore=[nn.BatchNorm2d])

    assert "conv Conv2d" in text
    assert "bn BatchNorm2d" not in text


def test_zero_structure_ignore_all_members_returns_empty_view_and_noop_prune() -> None:
    tracker = _dependency_built_linear_tracker()
    model = tracker.model

    view = tracker.view_zero_structures(ignore=[nn.Linear])
    result = tracker.prune_zero_structures(ignore=[nn.Linear])

    assert view.total_zero_units == 0
    assert result.pruned_units == 0
    assert result.view.total_zero_units == 0
    assert model.fc1.weight.shape == (3, 2)
    assert model.fc2.weight.shape == (1, 3)
