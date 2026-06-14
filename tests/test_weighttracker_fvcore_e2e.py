from __future__ import annotations

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


class CifarResNet20Block(nn.Module):
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
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class CifarResNet20(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_stage(16, 16, blocks=3, stride=1)
        self.layer2 = self._make_stage(16, 32, blocks=3, stride=2)
        self.layer3 = self._make_stage(32, 64, blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        *,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [CifarResNet20Block(in_channels, out_channels, stride)]
        layers.extend(
            CifarResNet20Block(out_channels, out_channels)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn(self.stem(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


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


class LargerTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp_in = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.mlp_out = nn.Linear(mlp_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp_out(self.activation(self.mlp_in(x)))
        return self.norm2(x + mlp_out)


class LargerTransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(48, 24)
        self.position_embed = nn.Embedding(12, 24)
        self.blocks = nn.ModuleList(
            [LargerTransformerBlock(24, 4, 48) for _ in range(2)]
        )
        self.head = nn.Linear(24, 7)

    def forward(self, token_ids: Tensor) -> Tensor:
        positions = torch.arange(
            token_ids.size(1),
            device=token_ids.device,
        ).unsqueeze(0)
        x = self.token_embed(token_ids) + self.position_embed(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])


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


def _effective_weight_tensors(module: nn.Module) -> tuple[torch.Tensor, ...]:
    if isinstance(module, nn.MultiheadAttention):
        in_proj_weight = getattr(module, "in_proj_weight", None)
        if isinstance(in_proj_weight, torch.Tensor):
            return (in_proj_weight,)

        weights = tuple(
            weight
            for weight in (
                getattr(module, "q_proj_weight", None),
                getattr(module, "k_proj_weight", None),
                getattr(module, "v_proj_weight", None),
            )
            if isinstance(weight, torch.Tensor)
        )
        if weights:
            return weights

    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return (weight,)

    return ()


def _active_weight_fraction(module: nn.Module) -> float:
    weights = _effective_weight_tensors(module)
    total_count = sum(weight.numel() for weight in weights)
    if total_count == 0:
        return 0.0
    nonzero_count = sum(int(weight.detach().ne(0).sum().item()) for weight in weights)
    return nonzero_count / total_count


def _expected_unstructured_bops_pr_module(
    tracker: WeightTracker,
    example_inputs,
    entries: tuple[tuple[str, nn.Module], ...] | None = None,
    bitrates: torch.Tensor | None = None,
) -> torch.Tensor:
    entries = (
        tuple(tracker._get_weighted_module_entries()) if entries is None else entries
    )
    by_module = _fvcore_by_module(tracker.model, example_inputs)
    bitrates = (
        tracker.get_calculation(CalcType.BITRATE_PR_MODULE)()
        if bitrates is None
        else bitrates
    ).view(-1, 2)
    values = [
        float(by_module[name])
        * _active_weight_fraction(module)
        * float(bitrates[index].prod())
        for index, (name, module) in enumerate(entries)
    ]
    return torch.tensor(values, dtype=bitrates.dtype, device=bitrates.device)


def _expected_bops_baseline_pr_module(
    tracker: WeightTracker,
    example_inputs,
    entries: tuple[tuple[str, nn.Module], ...] | None = None,
    bitrates: torch.Tensor | None = None,
) -> torch.Tensor:
    entries = (
        tuple(tracker._get_weighted_module_entries()) if entries is None else entries
    )
    by_module = _fvcore_by_module(tracker.model, example_inputs)
    bitrates = (
        tracker.get_calculation(CalcType.BITRATE_PR_MODULE)()
        if bitrates is None
        else bitrates
    )
    values = [
        float(by_module[name]) * 32 * 32
        for name, _ in entries
    ]
    return torch.tensor(values, dtype=bitrates.dtype, device=bitrates.device)


def _assert_named_tensor_values_close(
    actual: dict[str, torch.Tensor],
    expected_names: tuple[str, ...],
    expected_values: torch.Tensor,
) -> None:
    assert tuple(actual.keys()) == expected_names
    torch.testing.assert_close(torch.stack(tuple(actual.values())), expected_values)


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


def test_unstructured_bops_matches_fvcore_weighted_macs_for_resnet20() -> None:
    model = CifarResNet20().eval()
    model.stem.activation_bitrate = 8
    model.stem.weight_bitrate = 4
    model.layer2[0].conv1.bitrate = 6
    model.layer3[2].conv2.activation_bitrate = 4
    model.layer3[2].conv2.weight_bitrate = 2
    model.fc.bitrate = 8
    example_inputs = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        model.stem.weight[:, :, 0, 0] = 0
        model.layer1[1].conv2.weight[::2] = 0
        model.layer2[0].shortcut[0].weight[:, ::2] = 0
        model.layer3[2].conv1.weight[:, :, :, ::2] = 0
        model.fc.weight[:, ::3] = 0

    tracker = WeightTracker(
        model,
        example_inputs=example_inputs,
        root_module_types=[nn.Conv2d, nn.Linear],
    )
    metrics = tracker.create_tracker(
        TrackerType.UNSTRUCTURED_BOPS,
        log_total_bops=True,
        log_layerwise_stats=True,
        log_module_names=True,
        log_compression_rate=True,
    ).track()
    expected = _expected_unstructured_bops_pr_module(tracker, example_inputs)
    expected_baseline = _expected_bops_baseline_pr_module(tracker, example_inputs)
    expected_names = tuple(name for name, _ in tracker._get_weighted_module_entries())

    assert metrics["unstructured_bops_module_names"] == expected_names
    _assert_named_tensor_values_close(
        metrics["unstructured_bops_pr_module"],
        expected_names,
        expected,
    )
    _assert_named_tensor_values_close(
        metrics["unstructured_bops_baseline_pr_module"],
        expected_names,
        expected_baseline,
    )
    torch.testing.assert_close(metrics["unstructured_bops"], expected.sum())
    torch.testing.assert_close(
        metrics["unstructured_bops_baseline"],
        expected_baseline.sum(),
    )
    torch.testing.assert_close(
        metrics["unstructured_bops_compression"],
        1.0 - expected.sum() / expected_baseline.sum(),
    )
    torch.testing.assert_close(
        metrics["unstructured_bops_compression_rate"],
        metrics["unstructured_bops_compression"],
    )


def test_unstructured_bops_matches_fvcore_weighted_macs_for_transformer() -> None:
    model = LargerTransformerClassifier().eval()
    model.blocks[0].attn.activation_bitrate = 8
    model.blocks[0].attn.weight_bitrate = 4
    model.blocks[0].mlp_in.bitrate = 6
    model.blocks[1].attn.bitrate = 5
    model.blocks[1].mlp_out.activation_bitrate = 4
    model.blocks[1].mlp_out.weight_bitrate = 2
    model.head.activation_bitrate = 8
    model.head.weight_bitrate = 3
    token_ids = torch.randint(0, 48, (1, 12))

    with torch.no_grad():
        model.token_embed.weight[::5] = 0
        model.blocks[0].attn.in_proj_weight[::2] = 0
        model.blocks[0].mlp_in.weight[::3] = 0
        model.blocks[1].mlp_out.weight[:, ::4] = 0
        model.head.weight[:, ::2] = 0

    tracker = WeightTracker(
        model,
        example_inputs=token_ids,
        root_module_types=[nn.MultiheadAttention, nn.Linear],
        num_heads={
            block.attn: block.attn.num_heads
            for block in model.blocks
        },
    )
    unstructured_bops = tracker.create_tracker(
        TrackerType.UNSTRUCTURED_BOPS,
        ignore=[block.attn.out_proj for block in model.blocks],
        log_total_bops=True,
        log_layerwise_stats=True,
        log_module_names=True,
        log_compression_rate=True,
    )
    metrics = unstructured_bops.track()
    expected_names = metrics["unstructured_bops_module_names"]
    all_entries = dict(tracker._get_weighted_module_entries())
    expected_entries = tuple((name, all_entries[name]) for name in expected_names)
    bitrates = unstructured_bops.calc(CalcType.BITRATE_PR_MODULE)()
    expected = _expected_unstructured_bops_pr_module(
        tracker,
        token_ids,
        expected_entries,
        bitrates,
    )
    expected_baseline = _expected_bops_baseline_pr_module(
        tracker,
        token_ids,
        expected_entries,
        bitrates,
    )

    assert metrics["unstructured_bops_module_names"] == expected_names
    _assert_named_tensor_values_close(
        metrics["unstructured_bops_pr_module"],
        expected_names,
        expected,
    )
    _assert_named_tensor_values_close(
        metrics["unstructured_bops_baseline_pr_module"],
        expected_names,
        expected_baseline,
    )
    torch.testing.assert_close(metrics["unstructured_bops"], expected.sum())
    torch.testing.assert_close(
        metrics["unstructured_bops_baseline"],
        expected_baseline.sum(),
    )
    torch.testing.assert_close(
        metrics["unstructured_bops_compression"],
        1.0 - expected.sum() / expected_baseline.sum(),
    )
    torch.testing.assert_close(
        metrics["unstructured_bops_compression_rate"],
        metrics["unstructured_bops_compression"],
    )


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

    torch.testing.assert_close(axis_counts["attn"], torch.tensor([0.0, 3.0]))
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
