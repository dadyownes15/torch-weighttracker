from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn

import torch_weighttracker.calculations.calculations as calculation_impl
from torch_weighttracker.calculations import (
    CachedCalculation,
    CalcType,
    CalculationContext,
)
from torch_weighttracker.calculations.base import Calculation
from torch_weighttracker.canonical_units import (
    CanonicalMember,
    CanonicalUnitGroup,
    UnitKind,
    canonicalize_groups,
)
from torch_weighttracker.consumer_ignore import (
    ConsumerFilter,
    FilterItem,
    filter_canonical_members,
)
from torch_weighttracker.extractors.codeq_bitrate_extractor import (
    ModuleBitrateExtractor,
)
from torch_weighttracker.pruning.fake import (
    FakePruneUnitResult,
    fake_prune_canonical_unit,
    member_indices_for_unit,
)
from torch_weighttracker.reductions.builder import IndexSelection, SegmentSelection
from torch_weighttracker.regularizers import (
    RegularizerType,
    regularizer_class_for_type,
)
from torch_weighttracker.torch_pruning import ops
from torch_weighttracker.torch_pruning.dependency import DependencyGraph
from torch_weighttracker.torch_pruning.dependency.group import Group
from torch_weighttracker.trackers import (
    TrackerType,
    tracker_class_for_type,
)
from torch_weighttracker.trackers.base import (
    is_tracker_type_collection,
    normalize_tracker_types,
)


@dataclass(frozen=True)
class ZeroUnit:
    group_id: int
    unit_id: int
    canonical_id: int
    pruning_idxs: tuple[int, ...]


@dataclass(frozen=True)
class ZeroUnitGroup:
    group_id: int
    offset: int
    length: int
    zero_units: tuple[ZeroUnit, ...]


@dataclass(frozen=True)
class ZeroUnitView:
    groups: tuple[ZeroUnitGroup, ...]
    total_zero_units: int


@dataclass(frozen=True)
class PruneZeroUnitsResult:
    view: ZeroUnitView
    pruned_units: int
    dry_run: bool


@dataclass(frozen=True)
class PruneUnitResult:
    group_id: int
    unit_id: int
    pruning_idxs: tuple[int, ...]


class WeightTracker:
    def __init__(
        self,
        model: nn.Module,
        example_inputs=None,
        root_module_types=(ops.TORCH_CONV, ops.TORCH_LINEAR),
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
        post_prune_hooks: Iterable[Callable[["WeightTracker"], None]] = (),
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.num_heads = {} if num_heads is None else dict(num_heads)
        self.prune_dim = prune_dim
        self.prune_num_heads = prune_num_heads
        self.root_module_types = root_module_types
        self.forward_fn = forward_fn
        self.output_transform = output_transform
        self.unwrapped_parameters = _normalize_unwrapped_parameters(
            unwrapped_parameters
        )
        self.customized_pruners = customized_pruners
        self.post_prune_hooks = tuple(post_prune_hooks)
        self.dependency_graph = None

        self.ignored_layers = _expanded_ignored_layers(ignored_layers)
        self.ignored_params = [] if ignored_params is None else list(ignored_params)
        self.ignored_params.extend(
            item for item in self.ignored_layers if isinstance(item, nn.Parameter)
        )
        self.ignored_layers = [
            item for item in self.ignored_layers if isinstance(item, nn.Module)
        ]

        _validate_example_inputs_device(model, example_inputs)
        self.example_inputs = example_inputs

        self.groups = []
        self.canonical_groups = ()
        self.calculations = {}
        self._weighted_module_entries = None
        self._weighted_modules = None
        self._weighted_module_index = None
        self.regularizers = []
        self.trackers = []

        self._build_dependency_state()

    def _build_dependency_state(self) -> None:
        if self.example_inputs is None:
            self.dependency_graph = None
            self.groups = []
            self.canonical_groups = ()
            return

        self.dependency_graph = DependencyGraph().build_dependency(
            model=self.model,
            example_inputs=self.example_inputs,
            forward_fn=self.forward_fn,
            output_transform=self.output_transform,
            unwrapped_parameters=self.unwrapped_parameters,
            customized_pruners=self.customized_pruners,
            ignored_params=self.ignored_params,
        )

        groups = list(
            self.dependency_graph.get_all_groups(
                ignored_layers=self.ignored_layers,
                root_module_types=self.root_module_types,
            )
        )

        self.groups = []
        for group in groups:
            filtered_group = self._without_ignored_members(group)
            if filtered_group is not None:
                self.groups.append(filtered_group)

        self.canonical_groups = canonicalize_groups(
            self.groups,
            num_heads=self.num_heads,
            prune_dim=self.prune_dim,
            prune_num_heads=self.prune_num_heads,
        )

    def _uses_attention_view(self) -> bool:
        return bool(self.num_heads) and bool(self.prune_dim or self.prune_num_heads)

    def _is_attention_group(self, group) -> bool:
        for dep, _ in group:
            if dep.target.module in self.num_heads and (
                self.dependency_graph.is_out_channel_pruning_fn(dep.handler)
            ):
                return True

        return False

    def get_prune_unit(self, group_id: int, unit_id: int):
        group = self.canonical_groups[group_id]
        if unit_id < 0 or unit_id >= group.length:
            raise IndexError(
                f"unit_id {unit_id} is outside canonical group {group_id} "
                f"length {group.length}."
            )

        pruning_idxs = self._pruning_indices_for_unit(group_id, unit_id)

        pruning_group = self._get_pruning_group_for_indices(group_id, pruning_idxs)
        return pruning_group, pruning_idxs

    def prune_unit(self, group_id: int, unit_id: int) -> PruneUnitResult:
        pruning_group, pruning_idxs = self.get_prune_unit(group_id, unit_id)
        self._prepare_group_for_physical_prune(group_id, (unit_id,))
        pruning_group.prune()
        result = PruneUnitResult(
            group_id=group_id,
            unit_id=unit_id,
            pruning_idxs=tuple(int(index) for index in pruning_idxs),
        )
        self._refresh_after_physical_prune((result,))
        return result

    def fake_prune_unit(
        self,
        group_id: int,
        unit_id: int,
        *,
        prune_bias: bool = True,
    ) -> FakePruneUnitResult:
        result = fake_prune_canonical_unit(
            self.model,
            self.canonical_groups,
            group_id,
            unit_id,
            prune_bias=prune_bias,
        )
        # if result.zeroed_members > 0:
        #    self._invalidate_calculations()
        return result

    def view_zero_units(
        self,
        *,
        ignore: Iterable[FilterItem] = (),
    ) -> ZeroUnitView:
        filters = ConsumerFilter(ignore=ignore)
        filtered_groups = self._zero_detection_groups(filters=filters)
        visible_groups = tuple(
            (group, filtered_group)
            for group, filtered_group in zip(
                self.canonical_groups,
                filtered_groups,
                strict=True,
            )
            if len(filtered_group.members) > 0
        )
        if len(visible_groups) == 0:
            return ZeroUnitView(groups=(), total_zero_units=0)

        if filters:
            active_mask = self.ensure_calculations(
                (CalcType.UNIT_ACTIVE_MASK,),
                context=self._calculation_context(canonical_groups=filtered_groups),
            )[CalcType.UNIT_ACTIVE_MASK]()
        else:
            active_mask = self.get_calculation(CalcType.UNIT_ACTIVE_MASK)()
        zero_groups: list[ZeroUnitGroup] = []
        total_zero_units = 0

        for group, _ in visible_groups:
            zero_units: list[ZeroUnit] = []

            for unit_id in range(group.length):
                canonical_id = group.offset + unit_id
                if bool(active_mask[canonical_id].item()):
                    continue

                zero_units.append(
                    ZeroUnit(
                        group_id=group.group_id,
                        unit_id=unit_id,
                        canonical_id=canonical_id,
                        pruning_idxs=self._pruning_indices_for_unit(
                            group.group_id,
                            unit_id,
                        ),
                    )
                )

            if len(zero_units) == 0:
                continue

            total_zero_units += len(zero_units)
            zero_groups.append(
                ZeroUnitGroup(
                    group_id=group.group_id,
                    offset=group.offset,
                    length=group.length,
                    zero_units=tuple(zero_units),
                )
            )

        return ZeroUnitView(
            groups=tuple(zero_groups),
            total_zero_units=total_zero_units,
        )

    def prune_zero_units(
        self,
        *,
        dry_run: bool = False,
        ignore: Iterable[FilterItem] = (),
    ) -> PruneZeroUnitsResult:
        return self.prune_zero_structures(dry_run=dry_run, ignore=ignore)

    def view_zero_structures(
        self,
        *,
        ignore: Iterable[FilterItem] = (),
    ) -> ZeroUnitView:
        return self.view_zero_units(ignore=ignore)

    def prune_zero_structures(
        self,
        *,
        dry_run: bool = False,
        ignore: Iterable[FilterItem] = (),
    ) -> PruneZeroUnitsResult:
        view = self.view_zero_units(ignore=ignore)
        if dry_run or view.total_zero_units == 0:
            return PruneZeroUnitsResult(
                view=view,
                pruned_units=0,
                dry_run=dry_run,
            )

        events: list[PruneUnitResult] = []
        for zero_group in view.groups:
            pruning_idxs = tuple(
                sorted(
                    {
                        idx
                        for zero_unit in zero_group.zero_units
                        for idx in zero_unit.pruning_idxs
                    }
                )
            )
            pruning_group = self._get_pruning_group_for_indices(
                zero_group.group_id,
                pruning_idxs,
            )
            self._prepare_group_for_physical_prune(
                zero_group.group_id,
                tuple(zero_unit.unit_id for zero_unit in zero_group.zero_units),
            )
            pruning_group.prune()
            events.extend(
                PruneUnitResult(
                    group_id=zero_unit.group_id,
                    unit_id=zero_unit.unit_id,
                    pruning_idxs=zero_unit.pruning_idxs,
                )
                for zero_unit in zero_group.zero_units
            )

        self._refresh_after_physical_prune(events)
        return PruneZeroUnitsResult(
            view=view,
            pruned_units=view.total_zero_units,
            dry_run=False,
        )

    def _zero_detection_groups(
        self,
        *,
        filters: ConsumerFilter,
    ) -> tuple[CanonicalUnitGroup, ...]:
        if not filters:
            return tuple(self.canonical_groups)

        return filter_canonical_members(self.canonical_groups, filters)

    def _pruning_indices_for_unit(
        self,
        group_id: int,
        unit_id: int,
    ) -> tuple[int, ...]:
        group = self.canonical_groups[group_id]
        if len(group.members) == 0:
            raise ValueError(f"Canonical group {group_id} has no members.")
        if unit_id < 0 or unit_id >= group.length:
            raise IndexError(
                f"unit_id {unit_id} is outside canonical group {group_id} "
                f"length {group.length}."
            )

        return member_indices_for_unit(group.members[0], unit_id)

    def _get_pruning_group_for_indices(
        self,
        group_id: int,
        pruning_idxs: tuple[int, ...],
    ):
        if self.dependency_graph is None:
            raise ValueError("Physical pruning requires a dependency graph.")

        group = self.canonical_groups[group_id]
        if len(group.members) == 0:
            raise ValueError(f"Canonical group {group_id} has no members.")

        member = group.members[0]
        return self.dependency_graph.get_pruning_group(
            module=member.module,
            pruning_fn=member.handler,
            idxs=pruning_idxs,
        )

    def _prepare_group_for_physical_prune(
        self,
        group_id: int,
        unit_ids: Iterable[int],
    ) -> None:
        group = self.canonical_groups[group_id]
        if group.unit_kind != UnitKind.HEAD:
            return

        pruned_head_count = len({int(unit_id) for unit_id in unit_ids})
        if pruned_head_count == 0:
            return

        for member in group.members:
            if not isinstance(member.module, nn.MultiheadAttention):
                continue
            if member.num_heads is None:
                continue
            current_num_heads = int(self.num_heads.get(member.module, member.num_heads))
            post_prune_num_heads = max(1, current_num_heads - pruned_head_count)
            member.module.num_heads = post_prune_num_heads

    def _invalidate_calculations(self) -> None:
        self.calculations.clear()
        self._weighted_module_entries = None
        self._weighted_modules = None
        self._weighted_module_index = None
        self.trackers.clear()
        self.regularizers.clear()

    def _refresh_after_physical_prune(
        self,
        events: Iterable[PruneUnitResult],
    ) -> None:
        events = tuple(events)
        self._invalidate_calculations()
        self._update_logical_num_heads_after_prune(events)
        for hook in self.post_prune_hooks:
            hook(self)
        self._build_dependency_state()

    def _update_logical_num_heads_after_prune(
        self,
        events: Iterable[PruneUnitResult],
    ) -> None:
        pruned_units_by_group: dict[int, set[int]] = {}
        for event in events:
            pruned_units_by_group.setdefault(event.group_id, set()).add(event.unit_id)

        for group_id, unit_ids in pruned_units_by_group.items():
            group = self.canonical_groups[group_id]
            if group.unit_kind != UnitKind.HEAD:
                continue

            for member in group.members:
                if member.num_heads is None:
                    continue
                if member.pruning_indices_by_unit is None:
                    continue
                current_num_heads = int(
                    self.num_heads.get(member.module, member.num_heads)
                )
                self.num_heads[member.module] = max(
                    1,
                    current_num_heads - len(unit_ids),
                )

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
        context: CalculationContext | None = None,
        context_key: tuple | None = None,
        stack: tuple[CalcType, ...] = (),
    ):
        if context is None:
            context = self._calculation_context()
            context_key = None

        cache_key = self._calculation_cache_key(calculation_type, context_key)
        if cache_key not in self.calculations:
            self.calculations[cache_key] = self._build_calculation(
                calculation_type,
                context=context,
                context_key=context_key,
                stack=stack,
            )

        return self.calculations[cache_key]

    def get_calculation(self, calculation_type: CalcType | str):
        return self._get_calculation(CalcType(calculation_type))

    def _build_calculation(
        self,
        calculation_type: CalcType,
        *,
        context: CalculationContext,
        context_key: tuple | None,
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

        if spec.requires_groups and len(context.canonical_groups) == 0:
            raise ValueError(
                f"{calculation_type.value} requires dependency groups. Pass groups "
                "when constructing WeightTracker."
            )

        next_stack = (*stack, calculation_type)
        dependencies = {
            dependency: self._get_calculation(
                dependency,
                context=context,
                context_key=context_key,
                stack=next_stack,
            )
            for dependency in spec.required_calculations
        }
        calculation = spec.create(context, dependencies)

        if spec.cache_constant:
            cached = CachedCalculation(calculation)
            cached.refresh_cache()
            return cached

        return calculation

    def view_structures(
        self,
        *,
        ignore: Iterable[FilterItem] = (),
    ):
        canonical_groups = self._zero_detection_groups(
            filters=ConsumerFilter(ignore=ignore),
        )
        visible_groups = tuple(
            group for group in canonical_groups if len(group.members) > 0
        )
        module_names = _module_name_map(self.model)
        total_units = sum(group.length for group in visible_groups)
        lines = [
            "CanonicalGroups",
            f"groups={len(visible_groups)} total_units={total_units}",
        ]

        for group in visible_groups:
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
                if member.calculation_source_indices is not None:
                    details.append(
                        "calculation_source="
                        f"{_format_indices(member.calculation_source_indices)}"
                    )
                if member.pruning_indices_by_unit is not None:
                    details.append(
                        f"pruning_layout={_enum_value(member.pruning_index_layout)}"
                    )
                attention_details = _format_attention_details(member)
                if attention_details:
                    details.append(attention_details)
                lines.append(f"    - {' '.join(details)}")

        return "\n".join(lines)

    def ensure_calculations(
        self,
        calculation_types: tuple[CalcType, ...],
        *,
        context: CalculationContext | None = None,
    ) -> dict[CalcType, Calculation]:
        context_key = None
        if context is None:
            context = self._calculation_context()
        else:
            context_key = self._calculation_context_key(context)

        calculations = {}
        for calculation_type in calculation_types:
            calculations[calculation_type] = self._get_calculation(
                calculation_type,
                context=context,
                context_key=context_key,
                stack=(),
            )
        return calculations

    def create_tracker(
        self,
        tracker_type: TrackerType | str | Iterable[TrackerType | str],
        *,
        include: Iterable[FilterItem] = (),
        ignore: Iterable[FilterItem] = (),
        **kwargs,
    ):
        """
        Create and register a metric tracker.

        Tracker types:
            TrackerType.STRUCTURED_BOPS / "structured_bops":
                Tracks active structured bit operations from active runtime MACs
                and per-module activation/weight bitrates. Default output
                includes only "structured_bops_compression".
            TrackerType.L2_NORM_DISTRIBUTION / "l2_norm_distribution":
                Tracks each canonical group's per-prune-unit L2 norm
                distribution. Output keys use
                "l2_norm_distribution/<group_name>".
            TrackerType.UNSTRUCTURED_SPARSITY / "unstructured_sparsity":
                Tracks exact zero-weight sparsity as a global weighted fraction
                plus per-layer fractions. Output includes
                "unstructured_sparsity" and "layers".
            TrackerType.NVIDIA_2_4_SPARSITY / "nvidia_2_4_sparsity":
                Tracks strict NVIDIA 2:4 sparsity over contiguous groups of
                four weights along each supported layer's reduction axis.
                Output includes
                "nvidia_2_4_sparsity/strict_block_fraction",
                "nvidia_2_4_sparsity/nvidia_eligible_block_fraction",
                "nvidia_2_4_sparsity/strict_layers",
                "nvidia_2_4_sparsity/nvidia_eligible_layers",
                "nvidia_2_4_sparsity/total_layers", and
                "nvidia_2_4_sparsity/tail_elements".
            TrackerType.GROUP_PRUNING_SUMMARY / "group_pruning_summary":
                Tracks flat W&B-friendly pruned unit and group-attributed
                pruned parameter counts. Output includes
                "group_pruning/pruned_units", "group_pruning/pruned_params",
                and per-group scalar keys under "group_pruning/groups/".

        Args:
            tracker_type: The tracker type enum or string value to create, or
                a list/iterable of tracker type enums or string values to
                create together. Single values return one tracker. Iterable
                values return a list of trackers in the requested order.
                Valid strings are "structured_bops", "l2_norm_distribution",
                "unstructured_sparsity", "nvidia_2_4_sparsity", and
                "group_pruning_summary".
            include: Optional module instances or module types to keep in this
                tracker's calculation context. Module instances include their
                descendants.
            ignore: Optional module instances or module types to remove from
                this tracker's calculation context. Ignore wins when a module
                matches both include and ignore. Trackers apply these filters
                to the modules they measure, so per-module metrics and names
                follow the filtered context.
            **kwargs: Tracker-specific options. When creating multiple
                trackers, kwargs are passed to each tracker. Unsupported kwargs
                raise TypeError from the tracker constructor.

        StructuredBOPs kwargs:
            log_total_bops (bool): Include active and baseline structured BOP
                totals. When log_layerwise_stats=True, also include
                per-module BOP dictionaries. Default: False.
            log_module_names (bool): Include "structured_bops_module_names",
                aligned with per-module metric dictionaries. Default: False.
            log_layerwise_stats (bool): Include per-module StructuredBOPs
                metric dictionaries. Adds
                "structured_bops_compression_rate_pr_module"; when
                log_total_bops=True, also adds "structured_bops_pr_module" and
                "structured_bops_baseline_pr_module". Default: False.
            log_compression_rate (bool): Include the legacy
                "structured_bops_compression_rate" alias, computed as
                1 - structured_bops / structured_bops_baseline in this
                tracker's calculation context. The baseline currently uses
                hard-coded 32-bit activation and weight bitrates. Default:
                False.

        UnstructuredSparsity output:
            "unstructured_sparsity": Global zero-weight fraction, computed as
                total zero weight elements divided by total weight elements.
            "layers": A dict mapping module names to per-module zero-weight
                fractions.

        L2NormDistribution and UnstructuredSparsity have no public
        tracker-specific kwargs.

        Nvidia24Sparsity kwargs:
            log_layerwise_stats (bool): Include flat per-module metrics under
                "nvidia_2_4_sparsity/layers/<module_name>/...". Default:
                False.

        Nvidia24Sparsity output:
            "nvidia_2_4_sparsity/strict_block_fraction": Fraction of complete
                4-value blocks with exactly two zeros.
            "nvidia_2_4_sparsity/nvidia_eligible_block_fraction": Fraction of
                complete 4-value blocks with at least two zeros, matching
                NVIDIA/TensorRT eligibility.
            "nvidia_2_4_sparsity/strict_layers": Number of supported layers
                whose complete blocks are all strict and that have no tail
                elements.
            "nvidia_2_4_sparsity/nvidia_eligible_layers": Number of supported
                layers whose complete blocks are all NVIDIA-eligible and that
                have no tail elements.
            "nvidia_2_4_sparsity/total_layers": Number of supported measured
                layers. Supported layers are Linear, Conv1d/2d/3d, and
                MultiheadAttention projection weights.
            "nvidia_2_4_sparsity/tail_elements": Count of reduction-axis
                elements not covered by complete 4-value blocks.

        GroupPruningSummary output:
            "group_pruning/pruned_units": Total pruned canonical units.
            "group_pruning/pruned_params": Total group-attributed pruned
                parameter footprint.
            "group_pruning/groups/<group_name>/pruned_units": Per-group pruned
                canonical unit count.
            "group_pruning/groups/<group_name>/pruned_params": Per-group
                group-attributed pruned parameter footprint.
        """
        is_collection = is_tracker_type_collection(tracker_type)
        tracker_types = normalize_tracker_types(tracker_type)

        if is_collection:
            return [
                self._create_single_tracker(
                    tracker_type_item,
                    include=include,
                    ignore=ignore,
                    **kwargs,
                )
                for tracker_type_item in tracker_types
            ]

        return self._create_single_tracker(
            tracker_types[0],
            include=include,
            ignore=ignore,
            **kwargs,
        )

    def _create_single_tracker(
        self,
        tracker_type: TrackerType,
        *,
        include: Iterable[FilterItem] = (),
        ignore: Iterable[FilterItem] = (),
        **kwargs,
    ):
        tracker_cls = tracker_class_for_type(tracker_type)
        context = tracker_cls.calculation_context(
            self,
            include=include,
            ignore=ignore,
            **kwargs,
        )
        if context is not None:
            self._validate_consumer_context(context)
        tracker_kwargs = tracker_cls.constructor_kwargs(
            self,
            context=context,
            **kwargs,
        )
        calculations = self.ensure_calculations(
            tracker_cls.required_calculations,
            context=context,
        )
        tracker = tracker_cls(calculations=calculations, **tracker_kwargs)
        self.trackers.append(tracker)
        return tracker

    def create_regularizer(
        self,
        regularizer_type: RegularizerType | str,
        *,
        include: Iterable[FilterItem] = (),
        ignore: Iterable[FilterItem] = (),
        **kwargs,
    ):
        """
        Create and register a regularizer.

        Args:
            regularizer_type: The regularizer type or string value to create.
            include: Optional module instances or module types to keep in this
                regularizer's calculation context. Module instances include
                their descendants.
            ignore: Optional module instances or module types to remove from
                this regularizer's calculation context. Ignore wins when a
                module matches both include and ignore.
            **kwargs: Regularizer-specific options. Unsupported kwargs raise
                TypeError from the regularizer constructor.
        """
        regularizer_type = RegularizerType(regularizer_type)
        regularizer_cls = regularizer_class_for_type(regularizer_type)
        context = regularizer_cls.calculation_context(
            self,
            include=include,
            ignore=ignore,
            **kwargs,
        )
        if context is not None:
            self._validate_consumer_context(context)
        calculations = self.ensure_calculations(
            regularizer_cls.required_calculations,
            context=context,
        )
        regularizer = regularizer_cls(calculations=calculations, **kwargs)
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
                "when constructing WeightTracker."
            )

    def _calculation_context(
        self,
        *,
        canonical_groups: Iterable[CanonicalUnitGroup] | None = None,
        weighted_modules: Iterable[nn.Module] | None = None,
    ) -> CalculationContext:
        canonical_groups = (
            tuple(self.canonical_groups)
            if canonical_groups is None
            else tuple(canonical_groups)
        )
        weighted_modules = (
            self._get_weighted_modules()
            if weighted_modules is None
            else tuple(weighted_modules)
        )

        return CalculationContext(
            model=self.model,
            canonical_groups=canonical_groups,
            device=self.device,
            dtype=self.dtype,
            weighted_modules=weighted_modules,
            weighted_module_index={
                module: index for index, module in enumerate(weighted_modules)
            },
            example_inputs=self.example_inputs,
            weighted_module_names=self._module_names_for_modules(weighted_modules),
        )

    def _validate_consumer_context(self, context: CalculationContext) -> None:
        if len(context.canonical_groups) > 0 and all(
            len(group.members) == 0 for group in context.canonical_groups
        ):
            raise ValueError(
                "Consumer filters removed all canonical members from the calculation "
                "context."
            )

        if len(self._get_weighted_modules()) > 0 and len(context.weighted_modules) == 0:
            raise ValueError(
                "Consumer filters removed all weighted modules from the calculation "
                "context."
            )

    def _calculation_cache_key(
        self,
        calculation_type: CalcType,
        context_key: tuple | None,
    ) -> CalcType | tuple[CalcType, tuple]:
        if context_key is None:
            return calculation_type
        return calculation_type, context_key

    def _calculation_context_key(
        self,
        context: CalculationContext,
    ) -> tuple:
        return (
            tuple(_canonical_group_key(group) for group in context.canonical_groups),
            tuple(id(module) for module in context.weighted_modules),
            context.device,
            context.dtype,
            None if context.example_inputs is None else id(context.example_inputs),
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

    def _module_names_for_modules(
        self,
        modules: Iterable[nn.Module],
    ) -> tuple[str, ...]:
        names_by_module = _module_name_map(self.model)
        return tuple(
            names_by_module.get(
                module,
                f"<unnamed:{module.__class__.__name__}>",
            )
            for module in modules
        )


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


def _validate_example_inputs_device(model: nn.Module, example_inputs) -> None:
    if example_inputs is None:
        return

    model_devices = _model_tensor_devices(model)
    if len(model_devices) == 0:
        return

    sorted_model_devices = _sorted_devices(model_devices)
    if len(sorted_model_devices) != 1:
        raise ValueError(
            "WeightTracker requires all model parameters and buffers to live on "
            "one device. Found model devices: "
            f"{_format_devices(sorted_model_devices)}."
        )

    example_devices = _example_input_devices(example_inputs)
    if len(example_devices) == 0:
        return

    model_device = sorted_model_devices[0]
    sorted_example_devices = _sorted_devices(example_devices)
    if any(device != model_device for device in sorted_example_devices):
        raise ValueError(
            "WeightTracker example_inputs must live on the same device as the "
            f"model. Model device: {model_device}; example_inputs device(s): "
            f"{_format_devices(sorted_example_devices)}."
        )


def _model_tensor_devices(model: nn.Module) -> set[torch.device]:
    devices = {parameter.device for parameter in model.parameters()}
    devices.update(buffer.device for buffer in model.buffers())
    return devices


def _example_input_devices(example_inputs) -> set[torch.device]:
    devices = set()

    def collect(value) -> None:
        if isinstance(value, torch.Tensor):
            devices.add(value.device)
            return

        if isinstance(value, Mapping):
            for item in value.values():
                collect(item)
            return

        if isinstance(value, tuple | list):
            for item in value:
                collect(item)

    collect(example_inputs)
    return devices


def _sorted_devices(devices: Iterable[torch.device]) -> tuple[torch.device, ...]:
    return tuple(sorted(devices, key=str))


def _format_devices(devices: Iterable[torch.device]) -> str:
    return ", ".join(str(device) for device in devices)


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
    if member.projection_out_features is not None:
        details.append(f"projection_out_features={member.projection_out_features}")
    if member.projection_in_features is not None:
        details.append(f"projection_in_features={member.projection_in_features}")
    if member.embed_dim is not None:
        details.append(f"embed_dim={member.embed_dim}")
    if member.num_heads is not None:
        details.append(f"num_heads={member.num_heads}")
    if member.head_dim is not None:
        details.append(f"head_dim={member.head_dim}")
    return " ".join(details)


def _canonical_group_key(group: CanonicalUnitGroup) -> tuple:
    return (
        group.group_id,
        group.offset,
        group.length,
        group.unit_kind,
        tuple(_canonical_member_key(member) for member in group.members),
    )


def _canonical_member_key(member: CanonicalMember) -> tuple:
    return (
        id(member.module),
        member.unit_axis,
        member.source_layout,
        _selection_key(member.destination),
        (
            None
            if member.calculation_source_indices is None
            else tuple(int(index) for index in member.calculation_source_indices)
        ),
        (
            None
            if member.pruning_indices_by_unit is None
            else tuple(
                tuple(int(index) for index in indices)
                for indices in member.pruning_indices_by_unit
            )
        ),
        member.projection_out_features,
        member.projection_in_features,
        member.pruning_index_layout,
        member.num_heads,
        member.head_dim,
    )


def _selection_key(selection: SegmentSelection | IndexSelection) -> tuple:
    if isinstance(selection, SegmentSelection):
        return "segment", int(selection.start), int(selection.length)
    if isinstance(selection, IndexSelection):
        return "index", tuple(int(index) for index in selection.indices)
    raise TypeError(f"Unsupported selection type: {type(selection).__name__}")
