from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping
from numbers import Real
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
import torch_weighttracker
from fvcore.nn import FlopCountAnalysis
from torch_weighttracker.calculations import CalcType


def compare_weighttracker_to_physical_pruning(
    model: nn.Module,
    example_inputs,
    pruning_ratio: float = 0.5,
    *,
    importance: Callable | None = None,
    ignored_layers: Iterable[nn.Module] = (),
    round_to: int | None = 1,
    num_heads: Mapping[nn.Module, int] | None = None,
    unwrapped_parameters: Iterable[tuple[nn.Parameter, int]] | None = None,
    tracker_kwargs: Mapping[str, Any] | None = None,
    pruner_kwargs: Mapping[str, Any] | None = None,
    patch_timm_attention: bool = True,
    print_output: bool = True,
) -> dict[str, Any]:
    """
    Compare WeightTracker active MACs on a fake-pruned model with fvcore MACs on
    a physically pruned model.

    The same BasePruner configuration is applied to two deep copies:
    - zeroed_model: dependency groups are replayed by zeroing weight slices
    - physical_model: dependency groups are applied with group.prune()

    Physical MACs are measured with fvcore.FlopCountAnalysis.
    """

    tracker_kwargs = {} if tracker_kwargs is None else dict(tracker_kwargs)
    pruner_kwargs = {} if pruner_kwargs is None else dict(pruner_kwargs)

    zeroed_model = copy.deepcopy(model).eval()
    physical_model = copy.deepcopy(model).eval()

    if patch_timm_attention:
        _patch_timm_attention_forward(zeroed_model)
        _patch_timm_attention_forward(physical_model)

    if num_heads is None:
        zeroed_num_heads = _infer_num_heads(zeroed_model)
        physical_num_heads = _infer_num_heads(physical_model)
    else:
        zeroed_num_heads = _map_module_keys_by_name(
            num_heads,
            source_model=model,
            target_model=zeroed_model,
        )
        physical_num_heads = _map_module_keys_by_name(
            num_heads,
            source_model=model,
            target_model=physical_model,
        )

    if unwrapped_parameters is None:
        zeroed_unwrapped = _infer_unwrapped_parameters(zeroed_model)
        physical_unwrapped = _infer_unwrapped_parameters(physical_model)
    else:
        zeroed_unwrapped = _map_parameter_keys_by_name(
            unwrapped_parameters,
            source_model=model,
            target_model=zeroed_model,
        )
        physical_unwrapped = _map_parameter_keys_by_name(
            unwrapped_parameters,
            source_model=model,
            target_model=physical_model,
        )

    tracker = torch_weighttracker.WeightTracker(
        zeroed_model,
        example_inputs=example_inputs,
        num_heads=zeroed_num_heads,
        unwrapped_parameters=zeroed_unwrapped,
        **tracker_kwargs,
    )

    zeroed_pruner = _make_base_pruner(
        zeroed_model,
        example_inputs,
        pruning_ratio=pruning_ratio,
        importance=importance,
        ignored_layers=_map_modules_by_name(
            ignored_layers,
            source_model=model,
            target_model=zeroed_model,
        ),
        round_to=round_to,
        num_heads=zeroed_num_heads,
        unwrapped_parameters=zeroed_unwrapped,
        pruner_kwargs=pruner_kwargs,
    )
    physical_pruner = _make_base_pruner(
        physical_model,
        example_inputs,
        pruning_ratio=pruning_ratio,
        importance=importance,
        ignored_layers=_map_modules_by_name(
            ignored_layers,
            source_model=model,
            target_model=physical_model,
        ),
        round_to=round_to,
        num_heads=physical_num_heads,
        unwrapped_parameters=physical_unwrapped,
        pruner_kwargs=pruner_kwargs,
    )

    masks: dict[nn.Parameter, torch.Tensor] = {}
    skipped_fake_prune_items: list[tuple[str, str, int]] = []
    for group in zeroed_pruner.step(interactive=True):
        _fake_prune_group(group, masks, skipped_fake_prune_items)

    for group in physical_pruner.step(interactive=True):
        group.prune()
    _update_attention_metadata_after_pruning(physical_model, physical_pruner)

    physical_modules = dict(physical_model.named_modules())
    zeroed_modules = dict(zeroed_model.named_modules())

    physical_analysis = FlopCountAnalysis(physical_model, example_inputs)
    physical_analysis = physical_analysis.unsupported_ops_warnings(False)
    physical_analysis = physical_analysis.uncalled_modules_warnings(False)
    physical_total_macs = float(physical_analysis.total())
    physical_macs_by_module = {
        name: float(value) for name, value in physical_analysis.by_module().items()
    }
    physical_params_by_module = {
        name: _parameter_count(module)
        for name, module in physical_model.named_modules()
    }

    names = tracker._calculation_context().weighted_module_names
    baseline_axes = tracker.get_calculation(CalcType.BASELINE_MODULE_AXES)()
    baseline_macs = tracker.get_calculation(CalcType.BASELINE_MACS_PR_MODULE)()
    active_macs = tracker.get_calculation(CalcType.ACTIVE_MACS_PR_MODULE)()
    active_units = tracker.get_calculation(CalcType.UNIT_ACTIVE_MASK)()
    axis_delta = tracker.get_calculation(CalcType.UNIT_DELTA_TO_MODULE_AXIS)(
        active_units
    ).view_as(baseline_axes)
    active_axes = baseline_axes + axis_delta
    active_axes_by_module = _axis_rows_by_module(active_axes, num_modules=len(names))

    wt_axes_by_name = {
        name: _display_axis_row(axes)
        for name, axes in zip(names, active_axes_by_module)
    }
    wt_active_macs_by_name = {
        name: float(value) for name, value in zip(names, active_macs)
    }
    wt_baseline_macs_by_name = {
        name: float(value) for name, value in zip(names, baseline_macs)
    }

    comparison_rows = []
    for name in names:
        zeroed_module = zeroed_modules.get(name)
        physical_module = physical_modules.get(name)
        wt_macs = wt_active_macs_by_name[name]
        physical_macs = physical_macs_by_module.get(name)
        ratio = None if physical_macs in (None, 0) else wt_macs / physical_macs
        wt_axes = wt_axes_by_name.get(name)
        zeroed_axes = _nonzero_weight_axes(zeroed_module)
        physical_axes = _module_axes(physical_module)

        note = ""
        if zeroed_axes != physical_axes:
            note = "fake prune != physical axes"
        elif wt_axes != zeroed_axes:
            note = "axis mapping mismatch"
        elif ratio is None:
            note = "no fvcore module MACs"
        elif abs(ratio - 1.0) > 0.05:
            note = "MAC mismatch"

        comparison_rows.append(
            {
                "module": name,
                "kind": _module_kind(physical_module),
                "wt_axes": wt_axes,
                "zeroed_axes": zeroed_axes,
                "physical_axes": physical_axes,
                "axes_match": zeroed_axes == physical_axes,
                "wt_baseline_macs": wt_baseline_macs_by_name.get(name),
                "wt_macs": wt_macs,
                "physical_macs": physical_macs,
                "ratio": ratio,
                "note": note,
            }
        )

    physical_rows = []
    for name, module in physical_model.named_modules():
        if not _is_report_module(module):
            continue
        physical_rows.append(
            {
                "module": name or "<root>",
                "kind": _module_kind(module),
                "physical_axes": _module_axes(module),
                "physical_macs": physical_macs_by_module.get(name, 0.0),
                "params": physical_params_by_module.get(name, 0),
            }
        )

    summary = _build_summary(comparison_rows, skipped_fake_prune_items)

    if print_output:
        _print_report(
            baseline_macs=baseline_macs,
            active_macs=active_macs,
            physical_total_macs=physical_total_macs,
            physical_total_params=_parameter_count(physical_model),
            physical_rows=physical_rows,
            comparison_rows=comparison_rows,
            summary=summary,
        )

    return {
        "zeroed_model": zeroed_model,
        "physical_model": physical_model,
        "tracker": tracker,
        "weighttracker_baseline_macs": baseline_macs,
        "weighttracker_active_macs": active_macs,
        "physical_analysis": physical_analysis,
        "physical_macs_by_module": physical_macs_by_module,
        "physical_rows": physical_rows,
        "comparison_rows": comparison_rows,
        "summary": summary,
    }


def _make_base_pruner(
    model: nn.Module,
    example_inputs,
    *,
    pruning_ratio: float,
    importance: Callable | None,
    ignored_layers: Iterable[nn.Module],
    round_to: int | None,
    num_heads: Mapping[nn.Module, int],
    unwrapped_parameters: Iterable[tuple[nn.Parameter, int]],
    pruner_kwargs: Mapping[str, Any],
):
    if importance is None:
        importance = tp.importance.GroupMagnitudeImportance(p=2)

    return tp.pruner.BasePruner(
        model,
        example_inputs,
        importance=importance,
        pruning_ratio=pruning_ratio,
        ignored_layers=list(ignored_layers),
        round_to=round_to,
        num_heads=dict(num_heads),
        unwrapped_parameters=list(unwrapped_parameters),
        **dict(pruner_kwargs),
    )


def _fake_prune_group(group, masks, skipped):
    with torch.no_grad():
        for dep, idxs in group:
            layer = dep.target.module
            pruning_fn = dep.handler
            handler = getattr(pruning_fn, "__name__", str(pruning_fn))

            if isinstance(layer, nn.Conv2d):
                if pruning_fn in (
                    tp.prune_conv_out_channels,
                    tp.prune_depthwise_conv_out_channels,
                ):
                    _mask_param(
                        layer.weight,
                        (idxs, slice(None), slice(None), slice(None)),
                        masks,
                    )
                    _mask_param(layer.bias, idxs, masks)
                elif pruning_fn in (
                    tp.prune_conv_in_channels,
                    tp.prune_depthwise_conv_in_channels,
                ):
                    _mask_param(
                        layer.weight,
                        (slice(None), idxs, slice(None), slice(None)),
                        masks,
                    )
                else:
                    skipped.append((type(layer).__name__, handler, len(idxs)))

            elif isinstance(layer, nn.Linear):
                if pruning_fn == tp.prune_linear_out_channels:
                    _mask_param(layer.weight, (idxs, slice(None)), masks)
                    _mask_param(layer.bias, idxs, masks)
                elif pruning_fn == tp.prune_linear_in_channels:
                    _mask_param(layer.weight, (slice(None), idxs), masks)
                else:
                    skipped.append((type(layer).__name__, handler, len(idxs)))

            elif isinstance(layer, nn.modules.batchnorm._BatchNorm):
                if pruning_fn in (
                    tp.prune_batchnorm_out_channels,
                    tp.prune_batchnorm_in_channels,
                ):
                    _mask_param(layer.weight, idxs, masks)
                    _mask_param(layer.bias, idxs, masks)
                    layer.running_mean[idxs] = 0
                    layer.running_var[idxs] = 1
                else:
                    skipped.append((type(layer).__name__, handler, len(idxs)))

            elif isinstance(layer, nn.LayerNorm):
                if pruning_fn in (
                    tp.prune_layernorm_out_channels,
                    tp.prune_layernorm_in_channels,
                ):
                    _mask_param(layer.weight, idxs, masks)
                    _mask_param(layer.bias, idxs, masks)
                else:
                    skipped.append((type(layer).__name__, handler, len(idxs)))

            elif isinstance(layer, nn.MultiheadAttention):
                _fake_prune_multihead_attention(layer, idxs, pruning_fn, masks)

            elif isinstance(layer, nn.Parameter):
                continue

            else:
                skipped.append((type(layer).__name__, handler, len(idxs)))


def _mask_param(param, index, masks):
    if param is None:
        return
    if param not in masks:
        masks[param] = torch.ones_like(param)
    masks[param][index] = 0
    param.mul_(masks[param])


def _fake_prune_multihead_attention(layer, idxs, pruning_fn, masks):
    if pruning_fn not in (
        tp.prune_multihead_attention_out_channels,
        tp.prune_multihead_attention_in_channels,
    ):
        return

    embed_dim = int(layer.embed_dim)
    repeated = list(idxs) + [i + embed_dim for i in idxs] + [
        i + 2 * embed_dim for i in idxs
    ]

    if layer.in_proj_weight is not None:
        _mask_param(layer.in_proj_weight, (repeated, slice(None)), masks)
        _mask_param(layer.in_proj_weight, (slice(None), idxs), masks)
    if layer.in_proj_bias is not None:
        _mask_param(layer.in_proj_bias, repeated, masks)

    if layer.q_proj_weight is not None:
        _mask_param(layer.q_proj_weight, (idxs, slice(None)), masks)
    if layer.k_proj_weight is not None:
        _mask_param(layer.k_proj_weight, (idxs, slice(None)), masks)
    if layer.v_proj_weight is not None:
        _mask_param(layer.v_proj_weight, (idxs, slice(None)), masks)

    _mask_param(layer.out_proj.weight, (idxs, slice(None)), masks)
    _mask_param(layer.out_proj.weight, (slice(None), idxs), masks)
    _mask_param(layer.out_proj.bias, idxs, masks)


def _patch_timm_attention_forward(model: nn.Module) -> None:
    try:
        import timm

        attention_cls = timm.models.vision_transformer.Attention
    except Exception:
        return

    for module in model.modules():
        if isinstance(module, attention_cls):
            module.forward = _relaxed_timm_attention_forward.__get__(
                module,
                attention_cls,
            )


def _relaxed_timm_attention_forward(self, x, attn_mask=None):
    batch, tokens, _channels = x.shape
    qkv = self.qkv(x).reshape(
        batch,
        tokens,
        3,
        self.num_heads,
        self.head_dim,
    ).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(batch, tokens, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _infer_num_heads(model: nn.Module) -> dict[nn.Module, int]:
    mapping: dict[nn.Module, int] = {}
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            mapping[module] = int(module.num_heads)
        elif hasattr(module, "qkv") and hasattr(module, "num_heads"):
            qkv = getattr(module, "qkv")
            if isinstance(qkv, nn.Module):
                mapping[qkv] = int(module.num_heads)
    return mapping


def _infer_unwrapped_parameters(model: nn.Module) -> tuple[tuple[nn.Parameter, int], ...]:
    candidates = []
    for path in (
        ("cls_token",),
        ("class_token",),
        ("pos_embed",),
        ("encoder", "pos_embedding"),
    ):
        value = _get_attr_path(model, path)
        if isinstance(value, nn.Parameter):
            candidates.append((value, -1))
    return tuple(candidates)


def _get_attr_path(obj, path):
    current = obj
    for name in path:
        if not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current


def _update_attention_metadata_after_pruning(model, pruner) -> None:
    for module in model.modules():
        if hasattr(module, "qkv") and hasattr(module, "num_heads"):
            qkv = getattr(module, "qkv")
            if qkv in getattr(pruner, "num_heads", {}):
                module.num_heads = pruner.num_heads[qkv]
                module.head_dim = qkv.out_features // (3 * module.num_heads)


def _map_modules_by_name(
    modules: Iterable[nn.Module],
    *,
    source_model: nn.Module,
    target_model: nn.Module,
) -> tuple[nn.Module, ...]:
    source_names = {module: name for name, module in source_model.named_modules()}
    target_modules = dict(target_model.named_modules())
    mapped = []
    for module in modules:
        name = source_names.get(module)
        if name is not None and name in target_modules:
            mapped.append(target_modules[name])
    return tuple(mapped)


def _map_module_keys_by_name(
    mapping: Mapping[nn.Module, int],
    *,
    source_model: nn.Module,
    target_model: nn.Module,
) -> dict[nn.Module, int]:
    source_names = {module: name for name, module in source_model.named_modules()}
    target_modules = dict(target_model.named_modules())
    mapped = {}
    for module, value in mapping.items():
        name = source_names.get(module)
        if name is not None and name in target_modules:
            mapped[target_modules[name]] = int(value)
    return mapped


def _map_parameter_keys_by_name(
    params: Iterable[tuple[nn.Parameter, int]],
    *,
    source_model: nn.Module,
    target_model: nn.Module,
) -> tuple[tuple[nn.Parameter, int], ...]:
    source_names = {param: name for name, param in source_model.named_parameters()}
    target_params = dict(target_model.named_parameters())
    mapped = []
    for param, dim in params:
        name = source_names.get(param)
        if name is not None and name in target_params:
            mapped.append((target_params[name], int(dim)))
    return tuple(mapped)


def _is_report_module(module):
    return isinstance(
        module,
        (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.Linear,
            nn.LayerNorm,
            nn.modules.batchnorm._BatchNorm,
            nn.MultiheadAttention,
        ),
    ) or module.__class__.__name__ in {"Attention", "WindowAttention"}


def _module_kind(module):
    if module is None:
        return None
    return type(module).__name__


def _axis_rows_by_module(axes: torch.Tensor, *, num_modules: int) -> torch.Tensor:
    axes = axes.reshape(-1)
    if axes.numel() == num_modules * 2:
        return axes.reshape(num_modules, 2)
    if axes.numel() == num_modules:
        return axes.reshape(num_modules, 1)
    raise ValueError(
        "Cannot map WeightTracker module axes to module rows: "
        f"got {axes.numel()} axis values for {num_modules} modules."
    )


def _display_axis_row(axes: torch.Tensor) -> list[int]:
    values = [float(value) for value in axes.reshape(-1)]
    return [int(round(value)) for value in values if value >= 0]


def _module_axes(module):
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return [int(module.in_channels), int(module.out_channels)]
    if isinstance(module, nn.Linear):
        return [int(module.in_features), int(module.out_features)]
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return [int(module.num_features)]
    if isinstance(module, nn.LayerNorm):
        value = int(module.normalized_shape[-1])
        return [value]
    if isinstance(module, nn.MultiheadAttention):
        return [int(module.embed_dim), int(module.embed_dim)]
    if hasattr(module, "qkv") and isinstance(getattr(module, "qkv"), nn.Linear):
        qkv = getattr(module, "qkv")
        return [int(qkv.in_features), int(qkv.out_features)]
    return None


def _nonzero_weight_axes(module):
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        if isinstance(module, nn.MultiheadAttention):
            weight = getattr(module, "in_proj_weight", None)
        if not isinstance(weight, torch.Tensor):
            return None

    active = weight.detach().ne(0)

    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        active_out = active.flatten(1).any(dim=1)
        permute_order = (1, 0, *range(2, active.ndim))
        active_in = active.permute(permute_order).flatten(1).any(dim=1)
        return [int(active_in.sum()), int(active_out.sum())]

    if isinstance(module, nn.Linear):
        active_out = active.any(dim=1)
        active_in = active.any(dim=0)
        return [int(active_in.sum()), int(active_out.sum())]

    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        count = int(active.reshape(-1).sum())
        return [count]

    if isinstance(module, nn.LayerNorm):
        count = int(active.reshape(-1).sum())
        return [count]

    if isinstance(module, nn.MultiheadAttention):
        active_out = active.any(dim=1)
        active_in = active.any(dim=0)
        embed_dim = int(module.embed_dim)
        return [
            int(active_in.sum()),
            int(active_out[:embed_dim].sum()),
        ]

    return None


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _build_summary(rows, skipped_fake_prune_items):
    axis_mismatches = [
        row for row in rows if row["zeroed_axes"] != row["physical_axes"]
    ]
    wt_axis_mismatches = [
        row
        for row in rows
        if row["zeroed_axes"] == row["physical_axes"]
        and row["wt_axes"] != row["zeroed_axes"]
    ]
    mac_mismatches = [
        row
        for row in rows
        if row["ratio"] is not None and abs(row["ratio"] - 1.0) > 0.05
    ]
    return {
        "fake_zero_axis_mismatches": axis_mismatches,
        "weighttracker_axis_mismatches": wt_axis_mismatches,
        "mac_mismatches": mac_mismatches,
        "skipped_fake_prune_items": sorted(set(skipped_fake_prune_items)),
    }


def _print_report(
    *,
    baseline_macs,
    active_macs,
    physical_total_macs,
    physical_total_params,
    physical_rows,
    comparison_rows,
    summary,
) -> None:
    print("Physical fvcore totals")
    print("  physical total MACs:", physical_total_macs)
    print("  physical params:", physical_total_params)
    print()
    _print_table(
        "Physical pruned model, fvcore by_module()",
        physical_rows,
        [
            ("module", "module"),
            ("kind", "kind"),
            ("physical_axes", "physical_axes"),
            ("physical_macs", "physical_macs"),
            ("params", "params"),
        ],
    )

    print()
    print("WeightTracker MAC totals")
    print("  baseline MACs direct sum:", float(baseline_macs.sum()))
    print("  active MACs direct sum:", float(active_macs.sum()))
    print()
    _print_table(
        "WeightTracker zeroed model vs fvcore physical model",
        comparison_rows,
        [
            ("module", "module"),
            ("kind", "kind"),
            ("wt_axes", "wt_axes"),
            ("zeroed_axes", "zeroed_axes"),
            ("physical_axes", "physical_axes"),
            ("axes_match", "axes_match"),
            ("wt_macs", "wt_macs"),
            ("physical_macs", "physical_macs"),
            ("ratio", "ratio"),
            ("note", "note"),
        ],
    )

    print()
    print("Diagnostic summary")
    print(
        "  fake-zero axes differ from physical axes:",
        len(summary["fake_zero_axis_mismatches"]),
    )
    print(
        "  WeightTracker axes differ despite matching fake/physical axes:",
        len(summary["weighttracker_axis_mismatches"]),
    )
    print("  MAC mismatches > 5%:", len(summary["mac_mismatches"]))
    print("  skipped fake-prune items:", summary["skipped_fake_prune_items"])


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Real):
        return f"{float(value):.4g}"
    return str(value)


def _print_table(title, rows, columns) -> None:
    print(title)
    if not rows:
        print("  <no rows>")
        return

    widths = []
    for key, label in columns:
        values = [label, *(_fmt(row.get(key)) for row in rows)]
        widths.append(max(len(value) for value in values))

    header = "  " + "  ".join(
        label.ljust(width) for (_, label), width in zip(columns, widths)
    )
    divider = "  " + "  ".join("-" * width for width in widths)
    print(header)
    print(divider)
    for row in rows:
        print(
            "  "
            + "  ".join(
                _fmt(row.get(key)).ljust(width)
                for (key, _), width in zip(columns, widths)
            )
        )
