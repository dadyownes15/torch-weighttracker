from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional


def _format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"{numerator}/{denominator} (0.00%)"
    return f"{numerator}/{denominator} ({100.0 * numerator / denominator:.2f}%)"


def _format_axis_stats(stats: "AxisSparsityStats") -> str:
    structures = _format_ratio(stats.removed, stats.total)
    params = _format_ratio(stats.removed_params, stats.total_params)
    return f"structures={structures}, params={params}"


def _format_zero_units(zero_units: tuple[tuple[int, ...], ...], limit: int = 4) -> str:
    if len(zero_units) == 0:
        return "-"
    preview = ", ".join(str(unit) for unit in zero_units[:limit])
    if len(zero_units) > limit:
        preview += f", ... +{len(zero_units) - limit} more"
    return preview


class StructureAxis(str, Enum):
    OUT = "out"
    IN = "in"
    PARAM = "param"
    HEAD = "head"
    HEAD_DIM = "head_dim"


@dataclass(frozen=True)
class AnalyzerConfig:
    example_inputs: object
    forward_fn: Optional[Callable[..., Any]] = None
    output_transform: Optional[Callable[..., Any]] = None
    root_module_types: tuple[type, ...] = ()
    ignored_layers: tuple[object, ...] = ()
    ignored_params: tuple[object, ...] = ()
    customized_pruners: Optional[dict[Any, Any]] = None
    unwrapped_parameters: Optional[list[Any]] = None
    num_heads: Optional[dict[Any, int]] = None
    in_channel_groups: Optional[dict[Any, int]] = None
    out_channel_groups: Optional[dict[Any, int]] = None
    prune_num_heads: bool = False
    prune_head_dims: bool = True
    verbose: bool = True


@dataclass(frozen=True)
class GroupMemberView:
    module: Any
    module_name: str
    handler: Callable[..., Any]
    axis: StructureAxis
    local_idxs: tuple[int, ...]
    root_idxs: tuple[Optional[int], ...]
    measurable: bool
    reason_unmeasurable: Optional[str] = None


@dataclass(frozen=True)
class PruneUnitView:
    unit_index: int
    root_indices: tuple[int, ...]


@dataclass(frozen=True)
class GroupView:
    group_id: str
    root_module: Any
    root_module_name: str
    root_handler: Callable[..., Any]
    axis: StructureAxis
    atomic_size: int
    channel_groups: int
    size: int
    members: tuple[GroupMemberView, ...]
    prune_units: tuple[PruneUnitView, ...]
    attention_modules: tuple[Any, ...] = ()


@dataclass(frozen=True)
class AxisSparsityStats:
    total: int
    removed: int
    active: int
    sparsity: float
    total_params: int = 0
    removed_params: int = 0
    active_params: int = 0

    def __str__(self) -> str:
        return _format_axis_stats(self)


@dataclass(frozen=True)
class LayerSparsityStats:
    module: Any
    module_name: str
    out_stats: Optional[AxisSparsityStats] = None
    in_stats: Optional[AxisSparsityStats] = None
    param_stats: Optional[AxisSparsityStats] = None
    head_stats: Optional[AxisSparsityStats] = None
    head_dim_stats: Optional[AxisSparsityStats] = None


@dataclass(frozen=True)
class GroupSparsityStats:
    group_id: str
    root_module: Any
    root_module_name: str
    axis: StructureAxis
    stats: AxisSparsityStats
    zero_prune_units: tuple[tuple[int, ...], ...]

    def __str__(self) -> str:
        return (
            f"{self.group_id} [{self.axis.value}] "
            f"{_format_axis_stats(self.stats)} "
            f"zero_units={_format_zero_units(self.zero_prune_units)}"
        )


@dataclass(frozen=True)
class StructuredSparsityReport:
    global_stats: AxisSparsityStats
    by_group: dict[str, GroupSparsityStats]
    by_layer: dict[str, LayerSparsityStats]
    unsupported_groups: tuple[str, ...]
    model_total_params: int = 0

    def __str__(self) -> str:
        lines = [
            "StructuredSparsityReport",
            (
                "  global: "
                f"structures={_format_ratio(self.global_stats.removed, self.global_stats.total)}, "
                f"prunable_params={_format_ratio(self.global_stats.removed_params, self.global_stats.total_params)}, "
                f"model_params={self.model_total_params}"
            ),
        ]
        if self.by_group:
            lines.append("  groups:")
            for group_id in sorted(self.by_group):
                group_stats = self.by_group[group_id]
                lines.append(
                    f"    - {group_stats.group_id} [{group_stats.axis.value}] "
                    f"{_format_axis_stats(group_stats.stats)} "
                    f"zero_units={_format_zero_units(group_stats.zero_prune_units)}"
                )
        else:
            lines.append("  groups: -")
        if self.unsupported_groups:
            lines.append(f"  unsupported_groups: {', '.join(self.unsupported_groups)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class UnstructuredSparsityReport:
    zero_params: int
    total_params: int
    sparsity: float

    def __str__(self) -> str:
        return (
            "UnstructuredSparsityReport"
            f"(params={_format_ratio(self.zero_params, self.total_params)})"
        )


@dataclass(frozen=True)
class ReferenceSnapshot:
    per_group_sizes: dict[str, int]
    per_group_layer_names: dict[str, str]
    per_group_axes: dict[str, StructureAxis]


@dataclass(frozen=True)
class ZeroStructureCandidate:
    group_id: str
    root_module: Any
    root_module_name: str
    root_handler: Callable[..., Any]
    axis: StructureAxis
    root_indices: tuple[int, ...]
    zero_prune_units: tuple[tuple[int, ...], ...]
    supported: bool
    reason: Optional[str] = None
    attention_modules: tuple[Any, ...] = ()


@dataclass(frozen=True)
class PruneZeroResult:
    candidates: tuple[ZeroStructureCandidate, ...]
    pruned_group_ids: tuple[str, ...]


@dataclass(frozen=True)
class StructureSliceView:
    label: str
    parameter_name: str
    parameter_shape: tuple[int, ...]
    touched_params: int


@dataclass(frozen=True)
class StructureMemberInspection:
    module: Any
    module_name: str
    handler_name: str
    axis: StructureAxis
    local_indices: tuple[int, ...]
    root_indices: tuple[int, ...]
    measurable: bool
    reason_unmeasurable: Optional[str] = None
    touched_params: int = 0
    touched_tensors: int = 0
    slices: tuple[StructureSliceView, ...] = ()


@dataclass(frozen=True)
class StructureUnitInspection:
    unit_index: int
    root_indices: tuple[int, ...]
    touched_params: int
    touched_tensors: int
    members: tuple[StructureMemberInspection, ...]


@dataclass(frozen=True)
class StructureGroupInspection:
    group_id: str
    root_module: Any
    root_module_name: str
    axis: StructureAxis
    atomic_size: int
    channel_groups: int
    size: int
    attention_modules: tuple[Any, ...]
    members: tuple[GroupMemberView, ...]
    units: tuple[StructureUnitInspection, ...]


@dataclass(frozen=True)
class StructureSummaryRow:
    group_id: str
    axis: StructureAxis
    root_module_name: str
    structure_count: int
    coupled_member_count: int
    coupled_member_names: tuple[str, ...]
    total_touched_params: int
    touched_params_min: int
    touched_params_max: int
    touched_tensors_min: int
    touched_tensors_max: int


@dataclass(frozen=True)
class StructureInspectionReport:
    groups: tuple[StructureGroupInspection, ...]
    summary_rows: tuple[StructureSummaryRow, ...]
    model_total_params: int
    include_bias: bool = False
