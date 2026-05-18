from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from torch_weighttracker import WeightTracker
from torch_weighttracker.calculations import CalcType
from torch_weighttracker.consumer_ignore import FilterItem
from torch_weighttracker.trackers import TrackerType

tp = pytest.importorskip(
    "torch_pruning",
    reason=(
        "dev-local torch-pruning parity tests require the external "
        "torch-pruning package"
    ),
)


class TinyResidualBlock(nn.Module):
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


class TinyResidualClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(8)
        self.stem_act = nn.ReLU()
        self.block1 = TinyResidualBlock(8, 8)
        self.block2 = TinyResidualBlock(8, 16, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(16, 5, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


@dataclass(frozen=True)
class PreparedModels:
    dense: nn.Module
    masked: nn.Module
    physical: nn.Module


@dataclass(frozen=True)
class TrackerSnapshot:
    module_names: tuple[str, ...]
    active_macs: torch.Tensor
    baseline_macs: torch.Tensor
    active_axes: torch.Tensor
    bitrates: torch.Tensor
    metrics: dict[str, Any]


@dataclass(frozen=True)
class TorchPruningOracle:
    total_macs: float
    layer_macs_by_name: dict[str, float]


def test_tiny_structured_bops_matches_filtered_torch_pruning_oracle() -> None:
    example_inputs = torch.randn(1, 3, 16, 16)
    prepared = _prepare_pruned_models(
        _tiny_bitrate_model,
        example_inputs,
        ignored_layer_names=("stem_conv",),
        round_to=2,
    )

    snapshot = _tracker_snapshot(
        prepared.masked,
        example_inputs,
        ignore=(nn.BatchNorm2d,),
    )
    dense_oracle = _torch_pruning_oracle(prepared.dense, example_inputs)
    physical_oracle = _torch_pruning_oracle(prepared.physical, example_inputs)

    _assert_matches_torch_pruning_scope(
        snapshot,
        physical_oracle=physical_oracle,
        dense_oracle=dense_oracle,
        physical_model=prepared.physical,
    )


def test_tiny_include_and_ignore_scopes_match_filtered_torch_pruning_oracle() -> None:
    example_inputs = torch.randn(1, 3, 16, 16)
    prepared = _prepare_pruned_models(
        TinyResidualClassifier,
        example_inputs,
        ignored_layer_names=("stem_conv", "head"),
        round_to=2,
    )
    dense_oracle = _torch_pruning_oracle(prepared.dense, example_inputs)
    physical_oracle = _torch_pruning_oracle(prepared.physical, example_inputs)

    included_block = _tracker_snapshot(
        prepared.masked,
        example_inputs,
        include=(prepared.masked.block2,),
        ignore=(nn.BatchNorm2d,),
    )
    assert set(included_block.module_names) == {
        "block2.conv1",
        "block2.conv2",
        "block2.downsample.0",
    }
    _assert_matches_torch_pruning_scope(
        included_block,
        physical_oracle=physical_oracle,
        dense_oracle=dense_oracle,
        physical_model=prepared.physical,
    )

    without_classifier = _tracker_snapshot(
        prepared.masked,
        example_inputs,
        ignore=(nn.BatchNorm2d, prepared.masked.head),
    )
    assert "head" not in without_classifier.module_names
    _assert_matches_torch_pruning_scope(
        without_classifier,
        physical_oracle=physical_oracle,
        dense_oracle=dense_oracle,
        physical_model=prepared.physical,
    )


def test_resnet18_matches_filtered_torch_pruning_oracle() -> None:
    torchvision_models = pytest.importorskip(
        "torchvision.models",
        reason="ResNet18 dev-local parity test requires torchvision.",
    )
    example_inputs = torch.randn(1, 3, 64, 64)

    def factory() -> nn.Module:
        return torchvision_models.resnet18(weights=None)

    prepared = _prepare_pruned_models(
        factory,
        example_inputs,
        ignored_layer_names=("conv1", "fc"),
        round_to=8,
    )
    snapshot = _tracker_snapshot(
        prepared.masked,
        example_inputs,
        ignore=(nn.BatchNorm2d,),
    )
    dense_oracle = _torch_pruning_oracle(prepared.dense, example_inputs)
    physical_oracle = _torch_pruning_oracle(prepared.physical, example_inputs)

    _assert_matches_torch_pruning_scope(
        snapshot,
        physical_oracle=physical_oracle,
        dense_oracle=dense_oracle,
        physical_model=prepared.physical,
    )

    raw_gap = physical_oracle.total_macs - _sum_named(
        physical_oracle,
        snapshot.module_names,
    )
    assert raw_gap >= 0


def _tiny_bitrate_model() -> nn.Module:
    model = TinyResidualClassifier()
    model.stem_conv.activation_bitrate = 8
    model.stem_conv.weight_bitrate = 4
    model.block1.conv1.bitrate = 6
    model.block2.conv2.activation_bitrate = 4
    model.block2.conv2.weight_bitrate = 3
    return model


def _prepare_pruned_models(
    factory: Callable[[], nn.Module],
    example_inputs,
    *,
    ignored_layer_names: Iterable[str],
    round_to: int,
    pruning_ratio: float = 0.5,
    seed: int = 0,
) -> PreparedModels:
    torch.manual_seed(seed)
    masked = factory().eval()
    dense = copy.deepcopy(masked).eval()
    physical = copy.deepcopy(masked).eval()

    _apply_masked_torch_pruning(
        masked,
        example_inputs,
        ignored_layer_names=ignored_layer_names,
        round_to=round_to,
        pruning_ratio=pruning_ratio,
    )
    _apply_physical_torch_pruning(
        physical,
        example_inputs,
        ignored_layer_names=ignored_layer_names,
        round_to=round_to,
        pruning_ratio=pruning_ratio,
    )
    return PreparedModels(dense=dense, masked=masked, physical=physical)


def _apply_masked_torch_pruning(
    model: nn.Module,
    example_inputs,
    *,
    ignored_layer_names: Iterable[str],
    round_to: int,
    pruning_ratio: float,
) -> None:
    pruner = _build_pruner(
        model,
        example_inputs,
        ignored_layer_names=ignored_layer_names,
        round_to=round_to,
        pruning_ratio=pruning_ratio,
    )
    masks: dict[nn.Parameter, torch.Tensor] = {}
    for group in pruner.step(interactive=True):
        _fake_prune_group(group, masks)


def _apply_physical_torch_pruning(
    model: nn.Module,
    example_inputs,
    *,
    ignored_layer_names: Iterable[str],
    round_to: int,
    pruning_ratio: float,
) -> None:
    pruner = _build_pruner(
        model,
        example_inputs,
        ignored_layer_names=ignored_layer_names,
        round_to=round_to,
        pruning_ratio=pruning_ratio,
    )
    pruner.step()


def _build_pruner(
    model: nn.Module,
    example_inputs,
    *,
    ignored_layer_names: Iterable[str],
    round_to: int,
    pruning_ratio: float,
):
    modules = dict(model.named_modules())
    ignored_layers = [modules[name] for name in ignored_layer_names]
    return tp.pruner.BasePruner(
        model,
        example_inputs,
        importance=tp.importance.GroupMagnitudeImportance(p=2),
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        round_to=round_to,
    )


def _fake_prune_group(
    group,
    masks: dict[nn.Parameter, torch.Tensor],
) -> None:
    with torch.no_grad():
        for dep, idxs in group:
            layer = dep.target.module
            pruning_fn = dep.handler
            indices = _as_indices(idxs)

            if isinstance(layer, nn.Conv2d):
                if _handler_is(
                    pruning_fn,
                    "prune_conv_out_channels",
                    "prune_depthwise_conv_out_channels",
                    "prune_out_channels",
                ):
                    _mask_param(
                        masks,
                        layer.weight,
                        (indices, slice(None), slice(None), slice(None)),
                    )
                    if layer.bias is not None:
                        _mask_param(masks, layer.bias, indices)
                    continue

                if _handler_is(
                    pruning_fn,
                    "prune_conv_in_channels",
                    "prune_depthwise_conv_in_channels",
                    "prune_in_channels",
                ):
                    _mask_param(
                        masks,
                        layer.weight,
                        (slice(None), indices, slice(None), slice(None)),
                    )
                    continue

                raise AssertionError(
                    "Unsupported Conv2d pruning handler: "
                    f"{getattr(pruning_fn, '__name__', pruning_fn)!r}"
                )

            if isinstance(layer, nn.Linear):
                if _handler_is(
                    pruning_fn,
                    "prune_linear_out_channels",
                    "prune_out_channels",
                ):
                    _mask_param(masks, layer.weight, (indices, slice(None)))
                    if layer.bias is not None:
                        _mask_param(masks, layer.bias, indices)
                    continue

                if _handler_is(
                    pruning_fn,
                    "prune_linear_in_channels",
                    "prune_in_channels",
                ):
                    _mask_param(masks, layer.weight, (slice(None), indices))
                    continue

                raise AssertionError(
                    "Unsupported Linear pruning handler: "
                    f"{getattr(pruning_fn, '__name__', pruning_fn)!r}"
                )

            if isinstance(layer, nn.modules.batchnorm._BatchNorm) and _handler_is(
                pruning_fn,
                "prune_batchnorm_out_channels",
                "prune_batchnorm_in_channels",
                "prune_out_channels",
                "prune_in_channels",
            ):
                if layer.affine:
                    _mask_param(masks, layer.weight, indices)
                    _mask_param(masks, layer.bias, indices)
                if layer.track_running_stats:
                    layer.running_mean[indices] = 0
                    layer.running_var[indices] = 1
                continue

            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                raise AssertionError(
                    "Unsupported BatchNorm pruning handler: "
                    f"{getattr(pruning_fn, '__name__', pruning_fn)!r}"
                )


def _mask_param(
    masks: dict[nn.Parameter, torch.Tensor],
    param: nn.Parameter,
    index,
) -> None:
    if param not in masks:
        masks[param] = torch.ones_like(param)
    masks[param][index] = 0
    param.mul_(masks[param])


def _as_indices(idxs) -> list[int]:
    if hasattr(idxs, "tolist"):
        return [int(index) for index in idxs.tolist()]
    return [int(index) for index in idxs]


def _handler_is(handler, *names: str) -> bool:
    handler_name = getattr(handler, "__name__", None)
    return any(
        handler is getattr(tp, name, None) or handler_name == name for name in names
    )


def _tracker_snapshot(
    model: nn.Module,
    example_inputs,
    *,
    include: Iterable[FilterItem] = (),
    ignore: Iterable[FilterItem] = (),
) -> TrackerSnapshot:
    tracker = WeightTracker(model, example_inputs)
    structured_bops = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        include=include,
        ignore=ignore,
        log_total_bops=True,
        log_module_names=True,
        log_compression_rate=True,
    )
    metrics = structured_bops.track()
    active_macs_calc = structured_bops.calc(CalcType.ACTIVE_MACS_PR_MODULE)
    active_units = active_macs_calc.calc(CalcType.UNIT_ACTIVE_MASK)()
    baseline_axes = active_macs_calc.calc(CalcType.BASELINE_MODULE_AXES)()
    axis_delta = active_macs_calc.calc(CalcType.UNIT_DELTA_TO_MODULE_AXIS)(
        active_units
    ).view_as(baseline_axes)

    return TrackerSnapshot(
        module_names=tuple(metrics["structured_bops_module_names"]),
        active_macs=active_macs_calc(),
        baseline_macs=active_macs_calc.calc(CalcType.BASELINE_MACS_PR_MODULE)(),
        active_axes=baseline_axes + axis_delta,
        bitrates=structured_bops.calc(CalcType.BITRATE_PR_MODULE)().view(-1, 2),
        metrics=metrics,
    )


def _torch_pruning_oracle(model: nn.Module, example_inputs) -> TorchPruningOracle:
    model.eval()
    with torch.no_grad():
        total_macs, _, layer_macs, _ = tp.utils.count_ops_and_params(
            model,
            example_inputs,
            layer_wise=True,
        )
    names_by_module = {module: name for name, module in model.named_modules()}
    layer_macs_by_name = {
        names_by_module[module]: float(macs)
        for module, macs in layer_macs.items()
        if module in names_by_module
    }
    layer_macs_by_name = _subtract_bias_ops(
        model,
        example_inputs,
        layer_macs_by_name,
    )
    return TorchPruningOracle(
        total_macs=float(total_macs),
        layer_macs_by_name=layer_macs_by_name,
    )


def _subtract_bias_ops(
    model: nn.Module,
    example_inputs,
    layer_macs_by_name: dict[str, float],
) -> dict[str, float]:
    output_numels = _output_numels_for_biased_modules(model, example_inputs)
    adjusted = dict(layer_macs_by_name)
    for name, output_numel in output_numels.items():
        if name in adjusted:
            adjusted[name] -= float(output_numel)
    return adjusted


def _output_numels_for_biased_modules(
    model: nn.Module,
    example_inputs,
) -> dict[str, int]:
    output_numels: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output) -> None:
            output_numels[name] = int(output.numel())

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            _forward(model, example_inputs)
    finally:
        for handle in handles:
            handle.remove()

    return output_numels


def _forward(model: nn.Module, example_inputs):
    if isinstance(example_inputs, tuple):
        return model(*example_inputs)
    if isinstance(example_inputs, dict):
        return model(**example_inputs)
    return model(example_inputs)


def _assert_matches_torch_pruning_scope(
    snapshot: TrackerSnapshot,
    *,
    physical_oracle: TorchPruningOracle,
    dense_oracle: TorchPruningOracle,
    physical_model: nn.Module,
) -> None:
    expected_active_macs = _mac_tensor_for_names(physical_oracle, snapshot.module_names)
    expected_baseline_macs = _mac_tensor_for_names(dense_oracle, snapshot.module_names)
    expected_active_bops = expected_active_macs * snapshot.bitrates.prod(dim=1)
    expected_baseline_bops = expected_baseline_macs * (32 * 32)

    torch.testing.assert_close(snapshot.active_macs, expected_active_macs)
    torch.testing.assert_close(snapshot.baseline_macs, expected_baseline_macs)
    torch.testing.assert_close(
        torch.stack(tuple(snapshot.metrics["structured_bops_pr_module"].values())),
        expected_active_bops,
    )
    torch.testing.assert_close(
        snapshot.metrics["structured_bops"],
        expected_active_bops.sum(),
    )
    torch.testing.assert_close(
        torch.stack(
            tuple(snapshot.metrics["structured_bops_baseline_pr_module"].values())
        ),
        expected_baseline_bops,
    )
    torch.testing.assert_close(
        snapshot.metrics["structured_bops_baseline"],
        expected_baseline_bops.sum(),
    )
    torch.testing.assert_close(
        snapshot.metrics["structured_bops_compression"],
        1.0 - expected_active_bops.sum() / expected_baseline_bops.sum(),
    )
    torch.testing.assert_close(
        snapshot.active_axes,
        _axis_tensor_for_names(physical_model, snapshot.module_names),
    )


def _mac_tensor_for_names(
    oracle: TorchPruningOracle,
    names: tuple[str, ...],
) -> torch.Tensor:
    missing = [name for name in names if name not in oracle.layer_macs_by_name]
    if missing:
        raise AssertionError(
            "Torch-Pruning did not report layer MACs for tracked modules: "
            + ", ".join(missing)
        )
    return torch.tensor([oracle.layer_macs_by_name[name] for name in names])


def _axis_tensor_for_names(model: nn.Module, names: tuple[str, ...]) -> torch.Tensor:
    modules = dict(model.named_modules())
    return torch.tensor([_module_axes(modules[name]) for name in names])


def _module_axes(module: nn.Module) -> tuple[float, float]:
    if isinstance(module, nn.Conv2d):
        return float(module.in_channels), float(module.out_channels)
    if isinstance(module, nn.Linear):
        return float(module.in_features), float(module.out_features)
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return float(module.num_features), float(module.num_features)
    raise AssertionError(f"Unsupported tracked module type: {type(module).__name__}")


def _sum_named(oracle: TorchPruningOracle, names: tuple[str, ...]) -> float:
    return sum(oracle.layer_macs_by_name[name] for name in names)
