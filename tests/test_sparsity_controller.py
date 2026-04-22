from __future__ import annotations

import torch
import torch.nn as nn

import torch_structure_analyser as tsa


class ConvBnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 1, bias=False)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn(self.conv(x))


class GroupedConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 1, bias=False)
        self.gconv = nn.Conv2d(4, 4, 3, padding=1, groups=2, bias=False)

    def forward(self, x):
        return self.gconv(self.conv(x))


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.ll1 = nn.Linear(2, 3)
        self.ll2 = nn.Linear(3, 3)
        self.ll3 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.ll1(x).relu()
        x = self.ll2(x).relu()
        return self.ll3(x)


def test_structured_sparsity_counts_zero_coupled_conv_bn_units():
    model = ConvBnNet()
    with torch.no_grad():
        model.conv.weight[[0, 2]] = 0
        model.bn.weight[[0, 2]] = 0

    controller = tsa.SparsityTracker(
        model,
        example_inputs=torch.randn(1, 4, 8, 8),
    )
    report = controller.structured_sparsity()

    assert report.global_stats.total == 4
    assert report.global_stats.removed == 2
    assert report.global_stats.total_params == 20
    assert report.global_stats.removed_params == 10
    assert report.model_total_params == 24
    assert report.by_group["conv:prune_out_channels"].zero_prune_units == ((0,), (2,))


def test_grouped_conv_input_units_are_measured_in_prune_space():
    model = GroupedConvNet()
    with torch.no_grad():
        model.conv.weight[[0, 2]] = 0
        model.gconv.weight[:, [0], :, :] = 0

    controller = tsa.SparsityTracker(
        model,
        example_inputs=torch.randn(1, 4, 8, 8),
    )
    report = controller.structured_sparsity()

    assert report.by_group["conv:prune_out_channels"].stats.total == 2
    assert report.by_group["conv:prune_out_channels"].stats.removed == 1
    assert report.by_group["conv:prune_out_channels"].zero_prune_units == ((0, 2),)


def test_group_lasso_returns_scalar_with_grad():
    model = ConvBnNet()
    controller = tsa.SparsityTracker(
        model,
        example_inputs=torch.randn(1, 4, 8, 8),
    )

    loss, named_terms = controller.group_lasso()
    loss.backward()

    assert loss.ndim == 0
    assert named_terms
    assert all(term.ndim == 0 for term in named_terms.values())
    assert model.conv.weight.grad is not None
    assert model.bn.weight.grad is not None


def test_structured_sparsity_string_is_human_readable():
    model = ConvBnNet()
    with torch.no_grad():
        model.conv.weight[[0, 2]] = 0
        model.bn.weight[[0, 2]] = 0

    controller = tsa.SparsityTracker(
        model,
        example_inputs=torch.randn(1, 4, 8, 8),
    )
    report = controller.structured_sparsity()

    text = str(report)

    assert "StructuredSparsityReport" in text
    assert "global: structures=2/4 (50.00%)" in text
    assert "prunable_params=10/20 (50.00%)" in text
    assert "model_params=24" in text
    assert "conv:prune_out_channels [out]" in text
    assert "zero_units=(0,), (2,)" in text


def test_unstructured_sparsity_string_is_human_readable():
    model = ConvBnNet()
    with torch.no_grad():
        model.conv.weight.zero_()

    controller = tsa.SparsityTracker(
        model,
        example_inputs=torch.randn(1, 4, 8, 8),
    )

    text = str(controller.unstructured_sparsity())

    assert text.startswith("UnstructuredSparsityReport(")
    assert "params=" in text
    assert "%" in text


def test_prune_zero_structures_prunes_coupled_zero_units():
    model = ConvBnNet()
    with torch.no_grad():
        model.conv.weight[[0, 2]] = 0
        model.bn.weight[[0, 2]] = 0

    controller = tsa.SparsityTracker(
        model,
        example_inputs=torch.randn(1, 4, 8, 8),
    )
    reference = controller.capture_reference()
    result = controller.prune_zero_structures()
    report = controller.structured_sparsity(reference=reference)

    assert "conv:prune_out_channels" in result.pruned_group_ids
    assert model.conv.out_channels == 2
    assert model.bn.num_features == 2
    assert report.global_stats.total == 4
    assert report.global_stats.removed == 2


def test_global_prunable_params_are_deduplicated_across_overlapping_groups():
    model = TinyMLP()
    with torch.no_grad():
        model.ll2.weight[2, :] = 0
        model.ll3.weight[:, 2] = 0

    controller = tsa.SparsityTracker(
        model,
        example_inputs=torch.randn(1, 2),
    )
    report = controller.structured_sparsity()

    assert report.global_stats.total_params == 18
    assert report.global_stats.removed_params == 4
    assert report.model_total_params == 25
    assert report.by_group["ll2:prune_out_channels"].stats.removed_params == 4


def test_zeroing_ll3_adds_only_new_prunable_params_once_overlap_is_accounted_for():
    model = TinyMLP()
    with torch.no_grad():
        model.ll2.weight[2, :] = 0
        model.ll3.weight[:, 2] = 0

    controller = tsa.SparsityTracker(
        model,
        example_inputs=torch.randn(1, 2),
    )
    initial_report = controller.structured_sparsity()

    with torch.no_grad():
        model.ll3.weight[0, :] = 0

    updated_report = controller.structured_sparsity()

    assert initial_report.global_stats.removed_params == 4
    assert updated_report.global_stats.removed_params == 6


def test_list_structures_returns_full_inventory_for_mlp():
    model = TinyMLP()
    controller = tsa.SparsityTracker(
        model,
        example_inputs=torch.randn(1, 2),
    )

    report = controller.list_structures()

    assert report.model_total_params == sum(parameter.numel() for parameter in model.parameters())
    group_ids = {group.group_id for group in report.groups}
    assert "ll2:prune_out_channels" in group_ids

    mlp_group = next(group for group in report.groups if group.group_id == "ll2:prune_out_channels")
    first_unit = mlp_group.units[0]

    assert first_unit.root_indices == (0,)
    assert {"ll2", "ll3"}.issubset({member.module_name for member in first_unit.members})


def test_format_structure_summary_and_details_are_human_readable():
    model = TinyMLP()
    controller = tsa.SparsityTracker(
        model,
        example_inputs=torch.randn(1, 2),
    )

    summary_text = controller.format_structure_summary()

    assert "StructureSummary" in summary_text
    assert "group_id" in summary_text
    assert "ll2:prune_out_channels" in summary_text
    assert "  members for ll2:prune_out_channels:" in summary_text
    assert "    - ll2" in summary_text
    assert "    - ll3" in summary_text
