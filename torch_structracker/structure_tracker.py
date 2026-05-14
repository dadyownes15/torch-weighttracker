from typing import List
import torch
import torch.nn as nn

from torch_structracker.calculations.base import Calculation
import torch_structracker.calculations.calculations as calculation_impl
from torch_structracker.calculations import (
    CalcType,
    CachedCalculation,
    CalculationContext,
)

from torch_structracker.canonical_units import CanonicalUnitGroup, canonicalize_groups
from torch_structracker.extractors.codeq_bitrate_extractor import ModuleBitrateExtractor
from torch_structracker.reductions.builder import IndexSelection, SegmentSelection
from torch_structracker.regularizers import (
    RegularizerType,
    regularizer_class_for_type,
)
from torch_structracker.torch_pruning.dependency import DependencyGraph
from torch_structracker.torch_pruning.dependency.group import Group
from torch_structracker.trackers import (
    TrackerType,
    tracker_class_for_type,
)


class StructureTracker:
    def __init__(
        self,
        model: nn.Module,
        example_inputs=None,
        groups=None,
        root_module_types=None,
        forward_fn=None,
        output_transform=None,
        unwrapped_parameters=None,
        customized_pruners=None,
        ignored_layers=None,
        ignored_params=None,
        num_heads=None,
        prune_dim=None,
        prune_num_heads=False,
        device=None,
        dtype=None,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.num_heads = {} if num_heads is None else dict(num_heads)
        self.prune_dim = prune_dim
        self.prune_num_heads = prune_num_heads
        self.root_module_types = root_module_types
        self.ignored_layers = _expanded_ignored_layers(ignored_layers)
        self.ignored_params = [] if ignored_params is None else list(ignored_params)
        self.dependency_graph = None

        self.example_inputs = example_inputs

        if groups is None:
            if example_inputs is None:
                if root_module_types is not None:
                    raise ValueError(
                        "Dependency graph construction requires both example_inputs "
                        "and root_module_types."
                    )
                self.groups = []
            else:
                self.groups = self._build_groups(
                    example_inputs=example_inputs,
                    root_module_types=root_module_types,
                    forward_fn=forward_fn,
                    output_transform=output_transform,
                    unwrapped_parameters=unwrapped_parameters,
                    customized_pruners=customized_pruners,
                )
        else:
            self.groups = list(groups)
            
        if all(isinstance(group, CanonicalUnitGroup) for group in self.groups):
            self.canonical_groups = tuple(self.groups)
        else:
            self.canonical_groups = canonicalize_groups(
                self.groups,
                num_heads=self.num_heads,
                prune_dim=self.prune_dim,
                prune_num_heads=self.prune_num_heads,
            )

        self.calculations = {}
        self._weighted_module_entries = None
        self._weighted_modules = None
        self._weighted_module_index = None
        self.regularizers = []
        self.trackers = []

    def _build_groups(
        self,
        example_inputs,
        root_module_types=None,
        forward_fn=None,
        output_transform=None,
        unwrapped_parameters=None,
        customized_pruners=None,
    ):
        self.dependency_graph = DependencyGraph().build_dependency(
            model=self.model,
            example_inputs=example_inputs,
            forward_fn=forward_fn,
            output_transform=output_transform,
            unwrapped_parameters=_normalize_unwrapped_parameters(
                unwrapped_parameters
            ),
            customized_pruners=customized_pruners,
            ignored_params=self.ignored_params,
        )

        groups = list(
            self.dependency_graph.get_all_groups(
                ignored_layers=self.ignored_layers,
                # root_module_types=root_module_types,
            )
        )

        if self._uses_attention_view():
            groups = [group for group in groups if self._is_attention_group(group)]

        filtered_groups = []
        for group in groups:
            filtered_group = self._without_ignored_members(group)
            if filtered_group is not None:
                filtered_groups.append(filtered_group)

        return filtered_groups

    def _uses_attention_view(self) -> bool:
        return bool(self.num_heads) and bool(self.prune_dim or self.prune_num_heads)

    def _is_attention_group(self, group) -> bool:
        for dep, _ in group:
            if dep.target.module in self.num_heads and (
                self.dependency_graph.is_out_channel_pruning_fn(dep.handler)
            ):
                return True

        return False

    def _without_ignored_members(self, group):
        if len(self.ignored_layers) == 0:
            return group

        filtered_items = [
            item
            for item in group.items
            if item.dep.target.module not in self.ignored_layers
        ]

        if len(filtered_items) == 0:
            return None

        filtered_group = Group()
        filtered_group._group = list(filtered_items)
        filtered_group._DG = getattr(group, "_DG", None)
        return filtered_group
        
    def _get_calculation(
        self,
        calculation_type: CalcType,
        *,
        stack: tuple[CalcType, ...] = (),
    ):
        if calculation_type not in self.calculations:
            self.calculations[calculation_type] = self._create_calculation(
                calculation_type,
                stack=stack,
            )

        return self.calculations[calculation_type]

    def _create_calculation(
        self,
        calculation_type: CalcType,
        *,
        stack: tuple[CalcType, ...],
    ):
        if calculation_type in stack:
            cycle = (*stack, calculation_type)
            names = " -> ".join(item.value for item in cycle)
            raise ValueError(f"Circular calculation dependency detected: {names}")

        try:
            spec = calculation_impl.CALCULATION_SPECS[calculation_type]
        except KeyError as error:
            raise ValueError(
                f"Unknown calculation type: {calculation_type.value}"
            ) from error

        if spec.requires_groups:
            self._require_groups(calculation_type)

        next_stack = (*stack, calculation_type)
        dependencies = {
            dependency: self._get_calculation(dependency, stack=next_stack)
            for dependency in spec.required_calculations
        }
        calculation = spec.create(self._calculation_context(), dependencies)

        if spec.cache_constant:
            cached = CachedCalculation(calculation)
            cached.refresh_cache()
            return cached

        return calculation

    def view_structures(self):
        module_names = _module_name_map(self.model)
        total_units = sum(group.length for group in self.canonical_groups)
        lines = [
            "CanonicalGroups",
            f"groups={len(self.canonical_groups)} total_units={total_units}",
        ]

        for group in self.canonical_groups:
            start = group.offset
            stop = group.offset + group.length
            lines.extend(
                (
                    "",
                    (
                        f"- group {group.group_id}: units=[{start}:{stop}) "
                        f"length={group.length} kind={_enum_value(group.unit_kind)} "
                        f"members={len(group.members)}"
                    ),
                    "  members:",
                )
            )

            for member in group.members:
                module_name = module_names.get(
                    member.module,
                    f"<unnamed:{member.module.__class__.__name__}>",
                )
                details = [
                    module_name,
                    member.module.__class__.__name__,
                    f"axis={_enum_value(member.unit_axis)}",
                    f"layout={_enum_value(member.source_layout)}",
                    f"dest={_format_selection(member.destination)}",
                ]
                if member.source_indices is not None:
                    details.append(f"source={_format_indices(member.source_indices)}")
                attention_details = _format_attention_details(member)
                if attention_details:
                    details.append(attention_details)
                lines.append(f"    - {' '.join(details)}")

        return "\n".join(lines)

    def ensure_calculations(self, calculation_types: tuple[CalcType,...]) -> dict[CalcType, Calculation]:
        calculations = {}
        for calculation_type in calculation_types:
            calculations[calculation_type] = self._get_calculation(calculation_type)
        return calculations

    def create_tracker(self, tracker_type: TrackerType, **kwargs):
        """
        Tracker types:
            "structured_bops" - kwargs(ignore: List[nn.Module])
            
        """
        
        tracker_type = TrackerType(tracker_type)
        tracker_cls = tracker_class_for_type(tracker_type)
        calculations = self.ensure_calculations(tracker_cls.required_calculations)
        tracker = tracker_cls(calculations=calculations)
        self.trackers.append(tracker)
        return tracker

    def create_regularizer(self, regularizer_type: RegularizerType, ignore_modules: List[nn.Module] = []):
        regularizer_type = RegularizerType(regularizer_type)
        regularizer_cls = regularizer_class_for_type(regularizer_type)
        calculations = self.ensure_calculations(regularizer_cls.required_calculations)
        regularizer = regularizer_cls(calculations=calculations)
        self.regularizers.append(regularizer)
        return regularizer

    def track(self):
        metrics = {}
        for tracker in self.trackers:
            metrics.update(tracker.track())
        return metrics

    def _require_groups(self, calculation_type: CalcType) -> None:
        if len(self.canonical_groups) == 0:
            raise ValueError(
                f"{calculation_type.value} requires dependency groups. Pass groups "
                "when constructing StructTracker."
            )

    def _calculation_context(self) -> CalculationContext:
        return CalculationContext(
            model=self.model,
            canonical_groups=tuple(self.canonical_groups),
            device=self.device,
            dtype=self.dtype,
            weighted_modules=self._get_weighted_modules(),
            weighted_module_index=self._get_weighted_module_index(),
            example_inputs=self.example_inputs,
        )
        
    def _get_weighted_module_entries(self):
        if self._weighted_module_entries is None:
            self._weighted_module_entries = tuple(
                ModuleBitrateExtractor.weighted_modules(self.model)
            )
        return self._weighted_module_entries

    def _get_weighted_modules(self):
        if self._weighted_modules is None:
            self._weighted_modules = tuple(
                module for _, module in self._get_weighted_module_entries()
            )
        return self._weighted_modules

    def _get_weighted_module_index(self):
        if self._weighted_module_index is None:
            self._weighted_module_index = {
                module: index
                for index, module in enumerate(self._get_weighted_modules())
            }
        return self._weighted_module_index


def _expanded_ignored_layers(ignored_layers):
    if ignored_layers is None:
        return []

    expanded = []
    for layer in ignored_layers:
        if isinstance(layer, nn.Module):
            expanded.extend(list(layer.modules()))
        else:
            expanded.append(layer)

    return expanded


def _normalize_unwrapped_parameters(unwrapped_parameters):
    if unwrapped_parameters is None:
        return None

    if isinstance(unwrapped_parameters, dict):
        return list(unwrapped_parameters.items())

    return unwrapped_parameters


def _module_name_map(model: nn.Module) -> dict[nn.Module, str]:
    names = {}
    for name, module in model.named_modules():
        names[module] = name if name else "<root>"
    return names


def _enum_value(value) -> str:
    return str(getattr(value, "value", value))


def _format_selection(selection) -> str:
    if isinstance(selection, SegmentSelection):
        return f"[{selection.start}:{selection.start + selection.length})"
    if isinstance(selection, IndexSelection):
        return _format_indices(selection.indices)
    return str(selection)


def _format_indices(indices: tuple[int, ...], *, max_items: int = 8) -> str:
    if len(indices) <= max_items:
        return f"({', '.join(str(index) for index in indices)})"

    visible = ", ".join(str(index) for index in indices[:max_items])
    return f"({visible}, ... len={len(indices)})"


def _format_attention_details(member) -> str:
    details = []
    if member.embed_dim is not None:
        details.append(f"embed_dim={member.embed_dim}")
    if member.num_heads is not None:
        details.append(f"num_heads={member.num_heads}")
    if member.head_dim is not None:
        details.append(f"head_dim={member.head_dim}")
    return " ".join(details)
