from __future__ import annotations

import copy

import fvcore.nn
import torch
import torch.nn as nn
from torch import Tensor

from tests.fixtures_models import TinyTransformerClassifier
from torch_weighttracker import WeightTracker
from torch_weighttracker.calculations import CalcType
from torch_weighttracker.torch_pruning.pruner.function import (
    prune_batchnorm_in_channels,
    prune_batchnorm_out_channels,
    prune_conv_in_channels,
    prune_conv_out_channels,
    prune_depthwise_conv_in_channels,
    prune_depthwise_conv_out_channels,
    prune_layernorm_in_channels,
    prune_layernorm_out_channels,
    prune_linear_in_channels,
    prune_linear_out_channels,
    prune_multihead_attention_in_channels,
    prune_multihead_attention_out_channels,
)
from torch_weighttracker.trackers import TrackerType


class TinyResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + identity)


class TinyResNetClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(8)
        self.stem_act = nn.ReLU()
        self.block1 = TinyResNetBlock(8, 8)
        self.block2 = TinyResNetBlock(8, 16, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(16, 5)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class TinyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + 1e-6) * self.weight


class TinyRMSNormLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = TinyRMSNorm(8)
        self.proj = nn.Linear(8, 4, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.norm(x))


def _module_names(model: nn.Module) -> dict[nn.Module, str]:
    return {module: name for name, module in model.named_modules()}


def _group_containing(
    tracker: WeightTracker,
    module_name: str,
    handler,
):
    names = _module_names(tracker.model)
    for group in tracker.canonical_groups:
        for member in group.members:
            if names[member.module] == module_name and member.handler == handler:
                return group
    raise AssertionError(f"No canonical group contains {module_name}.")


def _axis_counts_by_module(tracker: WeightTracker) -> dict[str, torch.Tensor]:
    calculations = tracker.ensure_calculations(
        (CalcType.UNIT_ACTIVE_MASK, CalcType.UNITS_TO_MODULE_AXIS)
    )
    active = calculations[CalcType.UNIT_ACTIVE_MASK]()
    module_axis = calculations[CalcType.UNITS_TO_MODULE_AXIS](active).view(-1, 2)
    return {
        name: module_axis[index]
        for index, (name, _) in enumerate(tracker._get_weighted_module_entries())
    }


def _active_axes_by_module(tracker: WeightTracker) -> dict[str, torch.Tensor]:
    active = tracker.get_calculation(CalcType.UNIT_ACTIVE_MASK)()
    baseline_axes = tracker.get_calculation(CalcType.BASELINE_MODULE_AXES)()
    axis_delta = tracker.get_calculation(CalcType.UNIT_DELTA_TO_MODULE_AXIS)(
        active
    ).view_as(baseline_axes)
    active_axes = (baseline_axes + axis_delta).view(-1, 2)
    return {
        name: active_axes[index]
        for index, (name, _) in enumerate(tracker._get_weighted_module_entries())
    }


def _zero_group_unit(
    group,
    root_indices: tuple[int, ...],
    *,
    include_bias: bool = True,
) -> None:
    roots = set(root_indices)
    for member in group.members:
        raw_member = member.member
        local_indices = [
            int(local_index)
            for local_index, root_index in zip(
                raw_member.idxs,
                raw_member.root_idxs,
                strict=True,
            )
            if int(root_index) in roots
        ]
        if local_indices:
            _zero_member_slices(
                member.module,
                member.handler,
                local_indices,
                include_bias=include_bias,
            )


def _zero_member_slices(
    module: nn.Module,
    handler,
    indices: list[int],
    *,
    include_bias: bool,
) -> None:
    with torch.no_grad():
        if handler in {prune_conv_out_channels, prune_depthwise_conv_out_channels}:
            module.weight[indices, ...] = 0
            if include_bias and module.bias is not None:
                module.bias[indices] = 0
            return

        if handler in {prune_conv_in_channels, prune_depthwise_conv_in_channels}:
            module.weight[:, indices, ...] = 0
            return

        if handler == prune_linear_out_channels:
            module.weight[indices, :] = 0
            if include_bias and module.bias is not None:
                module.bias[indices] = 0
            return

        if handler == prune_linear_in_channels:
            module.weight[:, indices] = 0
            return

        if handler in {prune_batchnorm_out_channels, prune_batchnorm_in_channels}:
            if getattr(module, "affine", False):
                module.weight[indices] = 0
                if include_bias and module.bias is not None:
                    module.bias[indices] = 0
            return

        if handler in {prune_layernorm_out_channels, prune_layernorm_in_channels}:
            if getattr(module, "elementwise_affine", False):
                module.weight[indices] = 0
                if include_bias and module.bias is not None:
                    module.bias[indices] = 0
            return

        if handler in {
            prune_multihead_attention_out_channels,
            prune_multihead_attention_in_channels,
        }:
            _zero_attention_slices(module, indices, include_bias=include_bias)
            return

    raise AssertionError(f"Unhandled pruning handler: {handler.__name__}")


def _zero_attention_slices(
    attention: nn.MultiheadAttention,
    indices: list[int],
    *,
    include_bias: bool,
) -> None:
    embed_dim = attention.embed_dim
    repeated = list(indices)
    repeated += [index + embed_dim for index in indices]
    repeated += [index + 2 * embed_dim for index in indices]

    if attention.in_proj_weight is not None:
        attention.in_proj_weight[repeated, :] = 0
        attention.in_proj_weight[:, indices] = 0
    if include_bias and attention.in_proj_bias is not None:
        attention.in_proj_bias[repeated] = 0
    if attention.out_proj is not None:
        attention.out_proj.weight[indices, :] = 0
        attention.out_proj.weight[:, indices] = 0
        if include_bias and attention.out_proj.bias is not None:
            attention.out_proj.bias[indices] = 0


def _prune_group(group, root_indices: tuple[int, ...]) -> None:
    group.raw_group.prune(idxs=list(root_indices), record_history=False)


def _fvcore_by_module(model: nn.Module, example_inputs) -> dict[str, int]:
    analysis = fvcore.nn.FlopCountAnalysis(model, example_inputs)
    analysis = analysis.unsupported_ops_warnings(False)
    analysis = analysis.uncalled_modules_warnings(False)
    return dict(analysis.by_module())


def test_structured_bops_matches_fvcore_weighted_macs_for_dense_resnet() -> None:
    model = TinyResNetClassifier().eval()
    model.stem_conv.activation_bitrate = 8
    model.stem_conv.weight_bitrate = 4
    model.block1.conv1.bitrate = 6
    example_inputs = torch.randn(1, 3, 32, 32)
    tracker = WeightTracker(model, example_inputs)

    metrics = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        log_total_bops=True,
        log_layerwise_stats=True,
    ).track()
    bitrates = tracker.get_calculation(CalcType.BITRATE_PR_MODULE)().view(-1, 2)
    by_module = _fvcore_by_module(model, example_inputs)
    actual_pr_module = metrics["structured_bops_pr_module"]
    actual_values = torch.stack(tuple(actual_pr_module.values()))
    expected = torch.tensor(
        [
            float(by_module[name]) * float(bitrates[index].prod())
            for index, (name, _) in enumerate(tracker._get_weighted_module_entries())
        ],
        dtype=actual_values.dtype,
        device=actual_values.device,
    )

    assert tuple(actual_pr_module.keys()) == tuple(
        name for name, _ in tracker._get_weighted_module_entries()
    )
    torch.testing.assert_close(actual_values, expected)
    torch.testing.assert_close(metrics["structured_bops"], expected.sum())


def _conv2d_flops(module: nn.Conv2d, output_hw: tuple[int, int], batch_size: int = 1):
    kernel_h, kernel_w = module.kernel_size
    out_h, out_w = output_hw
    return (
        batch_size
        * out_h
        * out_w
        * module.out_channels
        * (module.in_channels // module.groups)
        * kernel_h
        * kernel_w
    )


def _linear_flops(
    module: nn.Linear,
    *,
    leading_elements: int,
):
    return leading_elements * module.in_features * module.out_features


def test_resnet_residual_prune_axis_counts_match_pruned_fvcore_modules() -> None:
    model = TinyResNetClassifier().eval()
    example_inputs = torch.randn(1, 3, 16, 16)
    tracker = WeightTracker(
        model,
        example_inputs=example_inputs,
        root_module_types=[nn.Conv2d, nn.Linear],
    )
    group = _group_containing(tracker, "block2.downsample.0", prune_conv_out_channels)

    _zero_group_unit(group, (0, 7))
    axis_counts = _axis_counts_by_module(tracker)

    torch.testing.assert_close(
        axis_counts["block2.downsample.0"],
        torch.tensor([8.0, 14.0]),
    )
    torch.testing.assert_close(axis_counts["block2.conv2"], torch.tensor([16.0, 14.0]))
    torch.testing.assert_close(axis_counts["head"], torch.tensor([14.0, 5.0]))

    _prune_group(group, (0, 7))

    assert model.block2.downsample[0].out_channels == 14
    assert model.block2.conv2.out_channels == 14
    assert model.head.in_features == 14
    assert model(example_inputs).shape == (1, 5)

    by_module = _fvcore_by_module(model, example_inputs)

    assert by_module["block2.downsample.0"] == _conv2d_flops(
        model.block2.downsample[0],
        output_hw=(8, 8),
    )
    assert by_module["block2.conv2"] == _conv2d_flops(
        model.block2.conv2,
        output_hw=(8, 8),
    )
    assert by_module["head"] == _linear_flops(model.head, leading_elements=1)


def test_transformer_mlp_prune_axis_counts_match_pruned_fvcore_modules() -> None:
    model = TinyTransformerClassifier().eval()
    token_ids = torch.randint(0, 32, (1, 8))
    tracker = WeightTracker(
        model,
        example_inputs=token_ids,
        root_module_types=[nn.Linear],
    )
    group = _group_containing(tracker, "mlp_in", prune_linear_out_channels)

    _zero_group_unit(group, (0, 7, 31))
    axis_counts = _axis_counts_by_module(tracker)

    torch.testing.assert_close(axis_counts["mlp_in"], torch.tensor([16.0, 29.0]))
    torch.testing.assert_close(axis_counts["mlp_out"], torch.tensor([29.0, 16.0]))

    _prune_group(group, (0, 7, 31))

    assert model.mlp_in.out_features == 29
    assert model.mlp_out.in_features == 29
    assert model(token_ids).shape == (1, 5)

    by_module = _fvcore_by_module(model, token_ids)

    assert by_module["mlp_in"] == _linear_flops(model.mlp_in, leading_elements=8)
    assert by_module["mlp_out"] == _linear_flops(model.mlp_out, leading_elements=8)
    assert by_module["head"] == _linear_flops(model.head, leading_elements=1)


def test_rmsnorm_has_feature_axes_and_zero_weighted_macs() -> None:
    model = TinyRMSNormLinear().eval()
    example_inputs = torch.randn(1, 4, 8)
    tracker = WeightTracker(
        model,
        example_inputs=example_inputs,
        unwrapped_parameters=[(model.norm.weight, -1)],
    )

    names = tracker._calculation_context().weighted_module_names
    norm_index = names.index("norm")
    baseline_axes = tracker.get_calculation(CalcType.BASELINE_MODULE_AXES)()
    baseline_macs = tracker.get_calculation(CalcType.BASELINE_MACS_PR_MODULE)()
    cost_axis_indices = tracker.get_calculation(CalcType.MODULE_AXIS_COST_INDICES)()
    active_macs = tracker.get_calculation(CalcType.ACTIVE_MACS_PR_MODULE)()
    norm_axis_start = norm_index * 2

    torch.testing.assert_close(
        baseline_axes[norm_axis_start : norm_axis_start + 2],
        torch.tensor([-1.0, 8.0]),
    )
    assert norm_axis_start not in cost_axis_indices.tolist()
    assert norm_axis_start + 1 in cost_axis_indices.tolist()
    torch.testing.assert_close(baseline_macs[norm_index], torch.tensor(0.0))
    torch.testing.assert_close(active_macs[norm_index], torch.tensor(0.0))


def test_transformer_attention_head_prune_matches_pruned_fvcore_modules() -> None:
    model = TinyTransformerClassifier().eval()
    token_ids = torch.randint(0, 32, (1, 8))
    tracker = WeightTracker(
        model,
        example_inputs=token_ids,
        root_module_types=[nn.MultiheadAttention, nn.Linear],
        num_heads={model.attn: model.attn.num_heads},
        prune_num_heads=True,
    )
    group = _group_containing(
        tracker,
        "attn",
        prune_multihead_attention_out_channels,
    )
    head_dim = model.attn.embed_dim // model.attn.num_heads

    _zero_group_unit(group, (0, 1, 2, 3))
    axis_counts = _axis_counts_by_module(tracker)

    torch.testing.assert_close(axis_counts["attn"], torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(axis_counts["mlp_in"], torch.tensor([3.0, 32.0]))
    torch.testing.assert_close(axis_counts["head"], torch.tensor([3.0, 5.0]))

    _prune_group(group, (0, 1, 2, 3))

    active_embed_dim = int(axis_counts["attn"][1].item()) * head_dim
    assert model.attn.embed_dim == active_embed_dim
    assert model.mlp_in.in_features == active_embed_dim
    assert model.mlp_out.out_features == active_embed_dim
    assert model.head.in_features == active_embed_dim
    assert model(token_ids).shape == (1, 5)

    by_module = _fvcore_by_module(model, token_ids)

    assert by_module["attn"] == 4 * token_ids.shape[1] * active_embed_dim**2
    assert by_module["mlp_in"] == _linear_flops(model.mlp_in, leading_elements=8)
    assert by_module["mlp_out"] == _linear_flops(model.mlp_out, leading_elements=8)
    assert by_module["head"] == _linear_flops(model.head, leading_elements=1)


def test_prune_num_heads_keeps_transformer_mlp_width_group_for_macs() -> None:
    torch.manual_seed(0)
    masked = TinyTransformerClassifier().eval()
    physical = copy.deepcopy(masked).eval()
    token_ids = torch.randint(0, 32, (1, 8))

    masked_tracker = _attention_head_tracker(masked, token_ids)
    masked_group = _group_containing(
        masked_tracker,
        "mlp_in",
        prune_linear_out_channels,
    )
    _zero_group_unit(masked_group, (0, 7, 31))

    physical_tracker = _attention_head_tracker(physical, token_ids)
    physical_group = _group_containing(
        physical_tracker,
        "mlp_in",
        prune_linear_out_channels,
    )
    _prune_group(physical_group, (0, 7, 31))

    active_axes = _active_axes_by_module(masked_tracker)
    module_names = tuple(
        name for name, _ in masked_tracker._get_weighted_module_entries()
    )
    active_macs = masked_tracker.get_calculation(CalcType.ACTIVE_MACS_PR_MODULE)()
    physical_macs = _mac_tensor_for_names(
        _fvcore_by_module(physical, token_ids),
        module_names,
        like=active_macs,
    )

    torch.testing.assert_close(active_axes["mlp_in"], torch.tensor([16.0, 29.0]))
    torch.testing.assert_close(active_axes["mlp_out"], torch.tensor([29.0, 16.0]))
    torch.testing.assert_close(active_macs, physical_macs)


def test_structured_bops_prune_num_heads_matches_mixed_transformer_pruning() -> None:
    torch.manual_seed(0)
    masked = TinyTransformerClassifier().eval()
    masked.attn.activation_bitrate = 8
    masked.attn.weight_bitrate = 4
    masked.mlp_in.bitrate = 6
    masked.mlp_out.activation_bitrate = 4
    masked.mlp_out.weight_bitrate = 3
    masked.head.bitrate = 5
    dense = copy.deepcopy(masked).eval()
    physical = copy.deepcopy(masked).eval()
    token_ids = torch.randint(0, 32, (1, 8))

    masked_tracker = _attention_head_tracker(masked, token_ids)
    masked_attention_group = _group_containing(
        masked_tracker,
        "attn",
        prune_multihead_attention_out_channels,
    )
    masked_mlp_group = _group_containing(
        masked_tracker,
        "mlp_in",
        prune_linear_out_channels,
    )
    _zero_group_unit(masked_attention_group, (0, 1, 2, 3))
    _zero_group_unit(masked_mlp_group, (0, 7, 31))

    physical_tracker = _attention_head_tracker(physical, token_ids)
    physical_attention_group = _group_containing(
        physical_tracker,
        "attn",
        prune_multihead_attention_out_channels,
    )
    physical_mlp_group = _group_containing(
        physical_tracker,
        "mlp_in",
        prune_linear_out_channels,
    )
    _prune_group(physical_attention_group, (0, 1, 2, 3))
    _prune_group(physical_mlp_group, (0, 7, 31))

    structured_bops = masked_tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        log_total_bops=True,
        log_module_names=True,
        log_layerwise_stats=True,
        log_compression_rate=True,
    )
    metrics = structured_bops.track()
    module_names = tuple(metrics["structured_bops_module_names"])
    active_axes = _active_axes_by_module(masked_tracker)
    active_macs = structured_bops.calc(CalcType.ACTIVE_MACS_PR_MODULE)()
    baseline_macs = structured_bops.calc(CalcType.BASELINE_MACS_PR_MODULE)()
    bitrates = structured_bops.calc(CalcType.BITRATE_PR_MODULE)().view(-1, 2)
    physical_macs = _mac_tensor_for_names(
        _fvcore_by_module(physical, token_ids),
        module_names,
        like=active_macs,
    )
    dense_macs = _mac_tensor_for_names(
        _fvcore_by_module(dense, token_ids),
        module_names,
        like=baseline_macs,
    )
    expected_active_bops = physical_macs * bitrates.prod(dim=1)
    expected_baseline_bops = dense_macs * (32 * 32)

    torch.testing.assert_close(active_axes["attn"], torch.tensor([12.0, 12.0]))
    torch.testing.assert_close(active_axes["mlp_in"], torch.tensor([12.0, 29.0]))
    torch.testing.assert_close(active_axes["mlp_out"], torch.tensor([29.0, 12.0]))
    assert "attn.out_proj" not in module_names
    torch.testing.assert_close(active_macs, physical_macs)
    torch.testing.assert_close(baseline_macs, dense_macs)
    torch.testing.assert_close(
        torch.stack(tuple(metrics["structured_bops_pr_module"].values())),
        expected_active_bops,
    )
    torch.testing.assert_close(metrics["structured_bops"], expected_active_bops.sum())
    torch.testing.assert_close(
        torch.stack(tuple(metrics["structured_bops_baseline_pr_module"].values())),
        expected_baseline_bops,
    )
    torch.testing.assert_close(
        metrics["structured_bops_baseline"],
        expected_baseline_bops.sum(),
    )


def test_multihead_attention_direct_macs_allows_uncalled_internal_out_proj() -> None:
    model = TinyTransformerClassifier().eval()
    token_ids = torch.randint(0, 32, (1, 8))
    tracker = _attention_head_tracker(model, token_ids)
    module_names = tracker._calculation_context().weighted_module_names
    out_proj_index = module_names.index("attn.out_proj")

    baseline_macs = tracker.get_calculation(CalcType.BASELINE_MACS_PR_MODULE)()

    assert "attn" in module_names
    torch.testing.assert_close(baseline_macs[out_proj_index], torch.tensor(0.0))


def test_structured_bops_attention_head_matches_pruned_fvcore_with_bitrates() -> None:
    torch.manual_seed(0)
    masked = TinyTransformerClassifier().eval()
    masked.attn.activation_bitrate = 8
    masked.attn.weight_bitrate = 4
    masked.mlp_in.bitrate = 6
    masked.mlp_out.activation_bitrate = 4
    masked.mlp_out.weight_bitrate = 3
    masked.head.bitrate = 5
    dense = copy.deepcopy(masked).eval()
    physical = copy.deepcopy(masked).eval()
    token_ids = torch.randint(0, 32, (1, 8))

    masked_tracker = _attention_head_tracker(masked, token_ids)
    masked_group = _group_containing(
        masked_tracker,
        "attn",
        prune_multihead_attention_out_channels,
    )
    _zero_group_unit(masked_group, (0, 1, 2, 3))

    physical_tracker = _attention_head_tracker(physical, token_ids)
    physical_group = _group_containing(
        physical_tracker,
        "attn",
        prune_multihead_attention_out_channels,
    )
    _prune_group(physical_group, (0, 1, 2, 3))

    structured_bops = masked_tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        log_total_bops=True,
        log_module_names=True,
        log_layerwise_stats=True,
        log_compression_rate=True,
    )
    metrics = structured_bops.track()
    module_names = tuple(metrics["structured_bops_module_names"])
    active_macs = structured_bops.calc(CalcType.ACTIVE_MACS_PR_MODULE)()
    baseline_macs = structured_bops.calc(CalcType.BASELINE_MACS_PR_MODULE)()
    bitrates = structured_bops.calc(CalcType.BITRATE_PR_MODULE)().view(-1, 2)
    physical_macs = _mac_tensor_for_names(
        _fvcore_by_module(physical, token_ids),
        module_names,
        like=active_macs,
    )
    dense_macs = _mac_tensor_for_names(
        _fvcore_by_module(dense, token_ids),
        module_names,
        like=baseline_macs,
    )
    expected_active_bops = physical_macs * bitrates.prod(dim=1)
    expected_baseline_bops = dense_macs * (32 * 32)

    assert "attn" in module_names
    assert "attn.out_proj" not in module_names
    torch.testing.assert_close(active_macs, physical_macs)
    torch.testing.assert_close(baseline_macs, dense_macs)
    torch.testing.assert_close(
        torch.stack(tuple(metrics["structured_bops_pr_module"].values())),
        expected_active_bops,
    )
    torch.testing.assert_close(metrics["structured_bops"], expected_active_bops.sum())
    torch.testing.assert_close(
        torch.stack(tuple(metrics["structured_bops_baseline_pr_module"].values())),
        expected_baseline_bops,
    )
    torch.testing.assert_close(
        metrics["structured_bops_baseline"],
        expected_baseline_bops.sum(),
    )


def _attention_head_tracker(
    model: TinyTransformerClassifier,
    token_ids: torch.Tensor,
) -> WeightTracker:
    return WeightTracker(
        model,
        example_inputs=token_ids,
        root_module_types=[nn.MultiheadAttention, nn.Linear],
        num_heads={model.attn: model.attn.num_heads},
        prune_num_heads=True,
    )


def _mac_tensor_for_names(
    by_module: dict[str, int],
    names: tuple[str, ...],
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    missing = [name for name in names if name not in by_module]
    if missing:
        raise AssertionError(
            "fvcore did not report MACs for tracked modules: " + ", ".join(missing)
        )
    return torch.tensor(
        [float(by_module[name]) for name in names],
        dtype=like.dtype,
        device=like.device,
    )
