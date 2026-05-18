import torch

from tests.test_calculation_specs import (
    FakeGroup,
    TinyQKVProjectionBlock,
    _member,
    _model_and_groups,
)
from torch_weighttracker import WeightTracker
from torch_weighttracker.canonical_units import canonicalize_groups
from torch_weighttracker.torch_pruning.pruner.function import (
    prune_linear_in_channels,
    prune_linear_out_channels,
)
from torch_weighttracker.trackers import TrackerType


def test_l2_norm_distribution_reports_exact_l2_values_pr_group() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, groups=groups)

    l2_tracker = tracker.create_tracker(TrackerType.L2_NORM_DISTRIBUTION)
    metrics = l2_tracker.track()

    assert tuple(metrics) == (
        "l2_norm_distribution/fc1:prune_out_channels",
        "l2_norm_distribution/fc2:prune_out_channels",
    )
    torch.testing.assert_close(
        metrics["l2_norm_distribution/fc1:prune_out_channels"],
        torch.tensor([torch.sqrt(torch.tensor(17.0)), 0.0, 7.0]),
    )
    torch.testing.assert_close(
        metrics["l2_norm_distribution/fc2:prune_out_channels"],
        torch.tensor([torch.sqrt(torch.tensor(52.0))]),
    )
    assert tracker.trackers == [l2_tracker]


def test_l2_norm_distribution_ignore_is_invariant_to_ignored_weights() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, groups=groups)
    l2_tracker = tracker.create_tracker(
        TrackerType.L2_NORM_DISTRIBUTION,
        ignore=[model.fc2],
    )

    before = l2_tracker.track()

    torch.testing.assert_close(
        before["l2_norm_distribution/fc1:prune_out_channels"],
        torch.tensor([1.0, 0.0, torch.sqrt(torch.tensor(13.0))]),
    )
    torch.testing.assert_close(
        before["l2_norm_distribution/fc2:prune_out_channels"],
        torch.tensor([0.0]),
    )

    with torch.no_grad():
        model.fc2.weight.add_(1000.0)

    after_ignored_change = l2_tracker.track()
    torch.testing.assert_close(
        after_ignored_change["l2_norm_distribution/fc1:prune_out_channels"],
        before["l2_norm_distribution/fc1:prune_out_channels"],
    )
    torch.testing.assert_close(
        after_ignored_change["l2_norm_distribution/fc2:prune_out_channels"],
        before["l2_norm_distribution/fc2:prune_out_channels"],
    )

    with torch.no_grad():
        model.fc1.weight.add_(1.0)

    after_tracked_change = l2_tracker.track()
    assert not torch.allclose(
        after_tracked_change["l2_norm_distribution/fc1:prune_out_channels"],
        before["l2_norm_distribution/fc1:prune_out_channels"],
    )


def test_l2_norm_distribution_include_is_invariant_to_excluded_weights() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, groups=groups)
    l2_tracker = tracker.create_tracker(
        TrackerType.L2_NORM_DISTRIBUTION,
        include=[model.fc1],
    )

    before = l2_tracker.track()

    torch.testing.assert_close(
        before["l2_norm_distribution/fc1:prune_out_channels"],
        torch.tensor([1.0, 0.0, torch.sqrt(torch.tensor(13.0))]),
    )
    torch.testing.assert_close(
        before["l2_norm_distribution/fc2:prune_out_channels"],
        torch.tensor([0.0]),
    )

    with torch.no_grad():
        model.fc2.weight.add_(1000.0)

    after_excluded_change = l2_tracker.track()
    torch.testing.assert_close(
        after_excluded_change["l2_norm_distribution/fc1:prune_out_channels"],
        before["l2_norm_distribution/fc1:prune_out_channels"],
    )
    torch.testing.assert_close(
        after_excluded_change["l2_norm_distribution/fc2:prune_out_channels"],
        before["l2_norm_distribution/fc2:prune_out_channels"],
    )

    with torch.no_grad():
        model.fc1.weight.add_(1.0)

    after_included_change = l2_tracker.track()
    assert not torch.allclose(
        after_included_change["l2_norm_distribution/fc1:prune_out_channels"],
        before["l2_norm_distribution/fc1:prune_out_channels"],
    )


def test_l2_norm_distribution_names_attention_head_dim_groups() -> None:
    model = TinyQKVProjectionBlock()
    groups = canonicalize_groups(
        (
            FakeGroup(
                _member(model.proj, prune_linear_in_channels, (0, 1, 2, 3)),
                _member(model.qkv, prune_linear_out_channels, tuple(range(12))),
            ),
        ),
        num_heads={model.qkv: 2},
        prune_dim=True,
    )
    tracker = WeightTracker(
        model,
        groups=groups,
        num_heads={model.qkv: 2},
        prune_dim=True,
    )

    metrics = tracker.create_tracker(TrackerType.L2_NORM_DISTRIBUTION).track()

    assert tuple(metrics) == (
        "l2_norm_distribution/proj:prune_in_channels:head_dim",
    )
