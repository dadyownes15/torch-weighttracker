import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from tests.test_calculation_specs import TinyLinearChain, _model_and_groups
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
        WeightTracker(model, example_inputs=example_inputs, groups=[])


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
    }
    torch.testing.assert_close(
        metrics["structured_bops_compression"],
        torch.tensor(1.0 - 96.0 / 9216.0),
    )
    assert tracker.trackers == [structured_bops]


@pytest.mark.parametrize(
    "tracker_types",
    (
        ["l2_norm_distribution", "group_pruning_summary"],
        [TrackerType.L2_NORM_DISTRIBUTION, TrackerType.GROUP_PRUNING_SUMMARY],
    ),
)
def test_create_tracker_accepts_tracker_type_lists(tracker_types) -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, groups=groups)

    created_trackers = tracker.create_tracker(tracker_types)
    metrics = tracker.track()

    assert tracker.trackers == created_trackers
    assert len(created_trackers) == 2
    assert "l2_norm_distribution/fc1:prune_out_channels" in metrics
    assert "group_pruning/pruned_units" in metrics


def test_invalid_tracker_type_lists_available_tracker_values() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, groups=groups)

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
    tracker = WeightTracker(model, groups=groups)

    regularizer = tracker.create_regularizer(RegularizerType.GROUP_LASSO)
    loss = regularizer()
    loss.backward()

    assert loss.ndim == 0
    assert model.fc1.weight.grad is not None
    assert model.fc2.weight.grad is not None
    assert tracker.regularizers == [regularizer]


def test_group_pruning_summary_reports_flat_unit_and_param_counts() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, groups=groups)

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
    tracker = WeightTracker(model, groups=groups)

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


def test_structured_bops_include_filters_weighted_modules_and_names() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

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
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

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
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

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
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

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
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

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


def test_structured_bops_layerwise_stats_are_opt_in() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

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


def test_structured_bops_total_bops_logging_is_opt_in() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

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
