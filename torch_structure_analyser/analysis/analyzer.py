from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

from .. import dependency, ops
from ..pruner import function
from .rules import build_default_rule_registry, infer_axis
from .types import (
    AnalyzerConfig,
    AxisSparsityStats,
    GroupMemberView,
    GroupSparsityStats,
    StructureGroupInspection,
    StructureInspectionReport,
    StructureMemberInspection,
    StructureSliceView,
    StructureSummaryRow,
    StructureUnitInspection,
    GroupView,
    LayerSparsityStats,
    PruneUnitView,
    ReferenceSnapshot,
    StructuredSparsityReport,
    StructureAxis,
    UnstructuredSparsityReport,
    ZeroStructureCandidate,
)


def build_atomic_prune_units(atomic_size: int) -> tuple[PruneUnitView, ...]:
    return tuple(
        PruneUnitView(unit_index=unit_index, root_indices=(unit_index,))
        for unit_index in range(atomic_size)
    )


def build_grouped_prune_units(atomic_size: int, channel_groups: int) -> tuple[PruneUnitView, ...]:
    if channel_groups <= 1:
        return build_atomic_prune_units(atomic_size)
    if atomic_size % channel_groups != 0:
        return build_atomic_prune_units(atomic_size)

    group_size = atomic_size // channel_groups
    return tuple(
        PruneUnitView(
            unit_index=unit_index,
            root_indices=tuple(unit_index + group_id * group_size for group_id in range(channel_groups)),
        )
        for unit_index in range(group_size)
    )


def build_head_prune_units(atomic_size: int, num_heads: int) -> tuple[PruneUnitView, ...]:
    if num_heads <= 0 or atomic_size % num_heads != 0:
        return build_atomic_prune_units(atomic_size)

    head_dim = atomic_size // num_heads
    return tuple(
        PruneUnitView(
            unit_index=head_id,
            root_indices=tuple(range(head_id * head_dim, (head_id + 1) * head_dim)),
        )
        for head_id in range(num_heads)
    )


class StructureAnalyzer:
    def __init__(self, model: nn.Module, config: AnalyzerConfig):
        self.model = model
        if len(config.root_module_types) == 0:
            config = replace(
                config,
                root_module_types=(ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM),
            )
        self.config = config
        self.num_heads = dict(config.num_heads or {})
        self.in_channel_groups = dict(config.in_channel_groups or {})
        self.out_channel_groups = dict(config.out_channel_groups or {})
        self._parameter_pruning_dims: dict[nn.Parameter, int] = {}
        self._group_views_cache: Optional[tuple[GroupView, ...]] = None
        self.DG: dependency.DependencyGraph | None = None
        self._rule_registry = build_default_rule_registry(self._parameter_pruning_dim)
        self.rebuild()

    @property
    def dg(self) -> dependency.DependencyGraph:
        return self.DG

    def rebuild(self) -> None:
        self.num_heads = dict(self.config.num_heads or {})
        self.in_channel_groups = dict(self.config.in_channel_groups or {})
        self.out_channel_groups = dict(self.config.out_channel_groups or {})
        self.DG = dependency.DependencyGraph().build_dependency(
            self.model,
            example_inputs=self.config.example_inputs,
            forward_fn=self.config.forward_fn,
            output_transform=self.config.output_transform,
            unwrapped_parameters=self.config.unwrapped_parameters,
            customized_pruners=self.config.customized_pruners,
            ignored_params=list(self.config.ignored_params),
            verbose=self.config.verbose,
        )

        self._parameter_pruning_dims = {
            item.parameters: item.pruning_dim for item in self.DG.unwrapped_parameters
        }
        self._detect_channel_groups()
        self._group_views_cache = None

    def iter_groups(self) -> tuple[GroupView, ...]:
        if self._group_views_cache is None:
            self._group_views_cache = tuple(self._normalize_all_groups())
        return self._group_views_cache

    def group_for(self, module, pruning_fn, idxs=None) -> GroupView:
        if idxs is None:
            if self.DG.is_out_channel_pruning_fn(pruning_fn):
                idxs = list(range(self.DG.get_out_channels(module)))
            else:
                idxs = list(range(self.DG.get_in_channels(module)))
        tp_group = self.DG.get_pruning_group(module, pruning_fn, idxs)
        return self._normalize_group_views(tp_group)[0]

    def group_views_for(self, module, pruning_fn, idxs=None) -> tuple[GroupView, ...]:
        if idxs is None:
            if self.DG.is_out_channel_pruning_fn(pruning_fn):
                idxs = list(range(self.DG.get_out_channels(module)))
            else:
                idxs = list(range(self.DG.get_in_channels(module)))
        tp_group = self.DG.get_pruning_group(module, pruning_fn, idxs)
        return self._normalize_group_views(tp_group)

    def capture_reference(self) -> ReferenceSnapshot:
        per_group_sizes = {}
        per_group_layer_names = {}
        per_group_axes = {}
        for group_view in self.iter_groups():
            per_group_sizes[group_view.group_id] = len(group_view.prune_units)
            per_group_layer_names[group_view.group_id] = group_view.root_module_name
            per_group_axes[group_view.group_id] = group_view.axis
        return ReferenceSnapshot(
            per_group_sizes=per_group_sizes,
            per_group_layer_names=per_group_layer_names,
            per_group_axes=per_group_axes,
        )

    def structured_sparsity(
        self,
        tol: float = 0.0,
        reference: ReferenceSnapshot | None = None,
        include_bias: bool = False,
    ) -> StructuredSparsityReport:
        current_group_stats: dict[str, GroupSparsityStats] = {}
        unsupported_groups: list[str] = []
        # A dict storing a mask for all tensors, where elements are true, if they are presenet in some structure
        global_prunable_masks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        global_removed_masks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        for group_view in self.iter_groups():
            zero_units: list[tuple[int, ...]] = []
            total_params = 0
            removed_params = 0
            supported_units = 0

            for unit in group_view.prune_units:
                tensors = self.unit_tensors(group_view, unit, include_bias=include_bias)
                parameter_masks = self.unit_parameter_masks(group_view, unit, include_bias=include_bias)
                if len(tensors) == 0:
                    continue
                supported_units += 1
                unit_param_count = sum(tensor.numel() for tensor in tensors)
                total_params += unit_param_count
                self._accumulate_parameter_masks(global_prunable_masks, parameter_masks)

                if self._all_zero(tensors, tol):
                    # Marks fully prunned units
                    zero_units.append(unit.root_indices)
                    removed_params += unit_param_count
                    self._accumulate_parameter_masks(global_removed_masks, parameter_masks)

            if supported_units == 0:
                unsupported_groups.append(group_view.group_id)
                continue

            active = supported_units - len(zero_units)
            total = supported_units
            if reference is not None:
                total = reference.per_group_sizes.get(group_view.group_id, total)
            removed = max(total - active, 0)
            stats = self._make_axis_stats(
                total=total,
                removed=removed,
                total_params=total_params,
                removed_params=removed_params,
            )
            current_group_stats[group_view.group_id] = GroupSparsityStats(
                group_id=group_view.group_id,
                root_module=group_view.root_module,
                root_module_name=group_view.root_module_name,
                axis=group_view.axis,
                stats=stats,
                zero_prune_units=tuple(zero_units),
            )

        if reference is not None:
            for group_id, total in reference.per_group_sizes.items():
                if group_id in current_group_stats or group_id in unsupported_groups:
                    continue
                current_group_stats[group_id] = GroupSparsityStats(
                    group_id=group_id,
                    root_module=None,
                    root_module_name=reference.per_group_layer_names[group_id],
                    axis=reference.per_group_axes[group_id],
                    stats=self._make_axis_stats(total=total, removed=total),
                    zero_prune_units=tuple(),
                )

        by_layer = self._aggregate_by_layer(current_group_stats.values())
        global_stats = self._aggregate_global(
            current_group_stats.values(),
            total_params=self._count_parameter_masks(global_prunable_masks),
            removed_params=self._count_parameter_masks(global_removed_masks),
        )
        return StructuredSparsityReport(
            global_stats=global_stats,
            by_group=current_group_stats,
            by_layer=by_layer,
            unsupported_groups=tuple(sorted(unsupported_groups)),
            model_total_params=sum(parameter.numel() for parameter in self.model.parameters()),
        )
    
    def calculate_bobs(self):
        # 


        pass

    def unstructured_sparsity(
        self,
        tol: float = 0.0,
        only_prunable: bool = False,
    ) -> UnstructuredSparsityReport:
        if only_prunable:
            parameters = self._prunable_parameters()
        else:
            parameters = list(self.model.parameters())

        total_params = 0
        zero_params = 0
        for parameter in parameters:
            total_params += parameter.numel()
            zero_params += torch.le(parameter.detach().abs(), tol).sum().item()

        sparsity = 0.0 if total_params == 0 else zero_params / total_params
        return UnstructuredSparsityReport(
            zero_params=zero_params,
            total_params=total_params,
            sparsity=sparsity,
        )
    

    def inspect_structures(
        self,
        include_bias: bool = False,
    ) -> StructureInspectionReport:
        groups: list[StructureGroupInspection] = []
        summary_rows: list[StructureSummaryRow] = []

        for group_view in self.iter_groups():
            unit_inspections = tuple(
                self._inspect_unit(group_view, unit, include_bias=include_bias)
                for unit in group_view.prune_units
            )
            groups.append(
                StructureGroupInspection(
                    group_id=group_view.group_id,
                    root_module=group_view.root_module,
                    root_module_name=group_view.root_module_name,
                    axis=group_view.axis,
                    atomic_size=group_view.atomic_size,
                    channel_groups=group_view.channel_groups,
                    size=group_view.size,
                    attention_modules=group_view.attention_modules,
                    members=group_view.members,
                    units=unit_inspections,
                )
            )

            touched_params = [unit.touched_params for unit in unit_inspections] or [0]
            touched_tensors = [unit.touched_tensors for unit in unit_inspections] or [0]
            coupled_member_names = tuple(
                dict.fromkeys(
                    member.module_name
                    for member in group_view.members
                    if member.measurable
                )
            )
            summary_rows.append(
                StructureSummaryRow(
                    group_id=group_view.group_id,
                    axis=group_view.axis,
                    root_module_name=group_view.root_module_name,
                    structure_count=len(unit_inspections),
                    coupled_member_count=sum(1 for member in group_view.members if member.measurable),
                    coupled_member_names=coupled_member_names,
                    total_touched_params=sum(touched_params),
                    touched_params_min=min(touched_params),
                    touched_params_max=max(touched_params),
                    touched_tensors_min=min(touched_tensors),
                    touched_tensors_max=max(touched_tensors),
                )
            )

        return StructureInspectionReport(
            groups=tuple(groups),
            summary_rows=tuple(summary_rows),
            model_total_params=sum(parameter.numel() for parameter in self.model.parameters()),
            include_bias=include_bias,
        )

    def zero_structure_candidates(
        self,
        tol: float = 0.0,
        include_bias: bool = False,
    ) -> tuple[ZeroStructureCandidate, ...]:
        candidates = []
        for group_view in self.iter_groups():
            zero_units = []
            for unit in group_view.prune_units:
                tensors = self.unit_tensors(group_view, unit, include_bias=include_bias)
                if len(tensors) == 0:
                    continue
                if self._all_zero(tensors, tol):
                    zero_units.append(unit.root_indices)
            if len(zero_units) == 0:
                continue
            root_indices = tuple(sorted({idx for unit in zero_units for idx in unit}))
            candidates.append(
                ZeroStructureCandidate(
                    group_id=group_view.group_id,
                    root_module=group_view.root_module,
                    root_module_name=group_view.root_module_name,
                    root_handler=group_view.root_handler,
                    axis=group_view.axis,
                    root_indices=root_indices,
                    zero_prune_units=tuple(zero_units),
                    supported=True,
                    attention_modules=group_view.attention_modules,
                )
            )
        axis_priority = {
            StructureAxis.HEAD: 0,
            StructureAxis.HEAD_DIM: 1,
            StructureAxis.OUT: 2,
            StructureAxis.IN: 3,
            StructureAxis.PARAM: 4,
        }
        candidates.sort(key=lambda candidate: (axis_priority.get(candidate.axis, 99), candidate.group_id))
        return tuple(candidates)

    def unit_tensors(
        self,
        group_view: GroupView,
        unit: PruneUnitView,
        include_bias: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        root_set = set(unit.root_indices)
        tensors = []
        for member in group_view.members:
            if not member.measurable:
                continue
            rule = self._rule_registry.resolve(member.handler)
            if rule is None:
                continue
            local_idxs = tuple(
                local_idx
                for local_idx, root_idx in zip(member.local_idxs, member.root_idxs)
                if root_idx in root_set
            )
            if len(local_idxs) == 0:
                continue
            member_tensors = rule.extract_tensors(
                member.module,
                local_idxs,
                include_bias=include_bias,
            )
            tensors.extend(tensor.reshape(-1) for tensor in member_tensors if tensor.numel() > 0)
        return tuple(tensors)

    def unit_parameter_masks(
        self,
        group_view: GroupView,
        unit: PruneUnitView,
        include_bias: bool = False,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        root_set = set(unit.root_indices)
        parameter_masks = []
        for member in group_view.members:
            if not member.measurable:
                continue
            rule = self._rule_registry.resolve(member.handler)
            if rule is None:
                continue
            local_idxs = tuple(
                local_idx
                for local_idx, root_idx in zip(member.local_idxs, member.root_idxs)
                if root_idx in root_set
            )
            if len(local_idxs) == 0:
                continue
            member_masks = rule.extract_masks(
                member.module,
                local_idxs,
                include_bias=include_bias,
            )
            parameter_masks.extend(
                (parameter, mask)
                for parameter, mask in member_masks
                if mask.numel() > 0 and mask.any().item()
            )
        return tuple(parameter_masks)

    def _inspect_unit(
        self,
        group_view: GroupView,
        unit: PruneUnitView,
        include_bias: bool,
    ) -> StructureUnitInspection:
        root_set = set(unit.root_indices)
        members = []

        for member in group_view.members:
            local_idxs = tuple(
                local_idx
                for local_idx, root_idx in zip(member.local_idxs, member.root_idxs)
                if root_idx in root_set
            )
            if len(local_idxs) == 0:
                continue
            members.append(
                self._inspect_member(
                    member,
                    local_idxs=local_idxs,
                    root_indices=tuple(sorted(root_set.intersection(member.root_idxs))),
                    include_bias=include_bias,
                )
            )

        parameter_masks = self.unit_parameter_masks(group_view, unit, include_bias=include_bias)
        touched_params = self._count_parameter_masks_by_parameter(parameter_masks)
        touched_tensors = sum(member.touched_tensors for member in members)
        return StructureUnitInspection(
            unit_index=unit.unit_index,
            root_indices=unit.root_indices,
            touched_params=touched_params,
            touched_tensors=touched_tensors,
            members=tuple(members),
        )

    def _inspect_member(
        self,
        member: GroupMemberView,
        local_idxs: tuple[int, ...],
        root_indices: tuple[int, ...],
        include_bias: bool,
    ) -> StructureMemberInspection:
        if not member.measurable:
            return StructureMemberInspection(
                module=member.module,
                module_name=member.module_name,
                handler_name=member.handler.__name__,
                axis=member.axis,
                local_indices=local_idxs,
                root_indices=root_indices,
                measurable=False,
                reason_unmeasurable=member.reason_unmeasurable,
            )

        member_masks = self._member_parameter_masks(member, local_idxs, include_bias=include_bias)
        slices = self._describe_member_slices(member, local_idxs, include_bias=include_bias)
        return StructureMemberInspection(
            module=member.module,
            module_name=member.module_name,
            handler_name=member.handler.__name__,
            axis=member.axis,
            local_indices=local_idxs,
            root_indices=root_indices,
            measurable=True,
            touched_params=self._count_parameter_masks_by_parameter(member_masks),
            touched_tensors=len(member_masks),
            slices=slices,
        )

    def _member_parameter_masks(
        self,
        member: GroupMemberView,
        local_idxs: tuple[int, ...],
        include_bias: bool,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if not member.measurable:
            return ()
        rule = self._rule_registry.resolve(member.handler)
        if rule is None:
            return ()
        member_masks = rule.extract_masks(
            member.module,
            local_idxs,
            include_bias=include_bias,
        )
        return tuple(
            (parameter, mask)
            for parameter, mask in member_masks
            if mask.numel() > 0 and mask.any().item()
        )

    def _describe_member_slices(
        self,
        member: GroupMemberView,
        local_idxs: tuple[int, ...],
        include_bias: bool,
    ) -> tuple[StructureSliceView, ...]:
        module = member.module
        idxs = tuple(sorted(set(int(idx) for idx in local_idxs)))
        slices: list[StructureSliceView] = []
        is_out = self.DG.is_out_channel_pruning_fn(member.handler)

        def add_slice(label: str, parameter_name: str, parameter, touched_params: int) -> None:
            slices.append(
                StructureSliceView(
                    label=label,
                    parameter_name=parameter_name,
                    parameter_shape=tuple(parameter.shape),
                    touched_params=touched_params,
                )
            )

        if isinstance(module, ops.TORCH_MHA):
            embed_dim = module.embed_dim
            repeated_idxs = list(idxs) + [idx + embed_dim for idx in idxs] + [idx + 2 * embed_dim for idx in idxs]
            if module.q_proj_weight is not None:
                add_slice("q_proj rows", "q_proj_weight", module.q_proj_weight, sum(module.q_proj_weight[idx, :].numel() for idx in idxs))
            if module.k_proj_weight is not None:
                add_slice("k_proj rows", "k_proj_weight", module.k_proj_weight, sum(module.k_proj_weight[idx, :].numel() for idx in idxs))
            if module.v_proj_weight is not None:
                add_slice("v_proj rows", "v_proj_weight", module.v_proj_weight, sum(module.v_proj_weight[idx, :].numel() for idx in idxs))
            if module.in_proj_weight is not None:
                row_touched = sum(module.in_proj_weight[idx, :].numel() for idx in repeated_idxs)
                col_touched = sum(module.in_proj_weight[:, idx].numel() for idx in idxs)
                add_slice("in_proj rows+cols", "in_proj_weight", module.in_proj_weight, row_touched + col_touched)
            if include_bias and module.in_proj_bias is not None:
                add_slice("in_proj_bias", "in_proj_bias", module.in_proj_bias, len(repeated_idxs))
            if module.out_proj is not None:
                row_touched = sum(module.out_proj.weight[idx, :].numel() for idx in idxs)
                col_touched = sum(module.out_proj.weight[:, idx].numel() for idx in idxs)
                add_slice("out_proj rows+cols", "out_proj.weight", module.out_proj.weight, row_touched + col_touched)
                if include_bias and module.out_proj.bias is not None:
                    add_slice("out_proj.bias", "out_proj.bias", module.out_proj.bias, len(idxs))
        elif isinstance(module, ops.TORCH_LINEAR):
            weight = module.weight
            if is_out:
                add_slice(
                    label=f"weight[{self._format_indices(idxs)}, :]",
                    parameter_name="weight",
                    parameter=weight,
                    touched_params=sum(weight[idx, :].numel() for idx in idxs),
                )
                if include_bias and getattr(module, "bias", None) is not None:
                    add_slice(
                        label=f"bias[{self._format_indices(idxs)}]",
                        parameter_name="bias",
                        parameter=module.bias,
                        touched_params=len(idxs),
                    )
            else:
                add_slice(
                    label=f"weight[:, {self._format_indices(idxs)}]",
                    parameter_name="weight",
                    parameter=weight,
                    touched_params=sum(weight[:, idx].numel() for idx in idxs),
                )
        elif isinstance(module, ops.TORCH_CONV):
            weight = module.weight
            if is_out:
                add_slice(
                    label=f"weight[{self._format_indices(idxs)}, ...]",
                    parameter_name="weight",
                    parameter=weight,
                    touched_params=sum(weight[idx, ...].numel() for idx in idxs),
                )
                if include_bias and getattr(module, "bias", None) is not None:
                    add_slice(
                        label=f"bias[{self._format_indices(idxs)}]",
                        parameter_name="bias",
                        parameter=module.bias,
                        touched_params=len(idxs),
                    )
            else:
                if getattr(module, "transposed", False):
                    touched = sum(weight[idx, ...].numel() for idx in idxs)
                else:
                    touched = sum(weight[:, idx, ...].numel() for idx in idxs)
                add_slice(
                    label=f"weight[:, {self._format_indices(idxs)}, ...]" if not getattr(module, "transposed", False) else f"weight[{self._format_indices(idxs)}, ...]",
                    parameter_name="weight",
                    parameter=weight,
                    touched_params=touched,
                )
        elif isinstance(module, (ops.TORCH_BATCHNORM, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if getattr(module, "weight", None) is not None:
                add_slice(
                    label=f"weight[{self._format_indices(idxs)}]",
                    parameter_name="weight",
                    parameter=module.weight,
                    touched_params=len(idxs),
                )
            if include_bias and getattr(module, "bias", None) is not None:
                add_slice(
                    label=f"bias[{self._format_indices(idxs)}]",
                    parameter_name="bias",
                    parameter=module.bias,
                    touched_params=len(idxs),
                )
        elif isinstance(module, nn.Embedding):
            weight = module.weight
            add_slice(
                label=f"weight[:, {self._format_indices(idxs)}]",
                parameter_name="weight",
                parameter=weight,
                touched_params=sum(weight[:, idx].numel() for idx in idxs),
            )
        elif isinstance(module, nn.Parameter):
            add_slice(
                label=f"param[{self._format_indices(idxs)}]",
                parameter_name=self._module_name(module),
                parameter=module,
                touched_params=len(idxs),
            )

        return tuple(slices)

    def _normalize_all_groups(self) -> Iterable[GroupView]:
        for group in self.DG.get_all_groups(
            ignored_layers=self._ignored_layers(),
            root_module_types=self.config.root_module_types,
        ):
            attention_group = self._downstream_node_as_root_if_attention(group)
            if attention_group is not None:
                group = attention_group
            yield from self._normalize_group_views(group)

    def _normalize_group_views(self, group) -> tuple[GroupView, ...]:
        root_module = group[0].dep.target.module
        root_handler = group[0].dep.handler
        root_axis = infer_axis(root_handler)
        is_attention, qkv_layers = self._is_atten_group(group)
        channel_groups = self._get_channel_groups(group)
        attention_modules = tuple(qkv_layers)
        atomic_size = len(group[0].root_idxs)

        members = []
        for i, (dep, idxs) in enumerate(group):
            handler = dep.handler
            rule = self._rule_registry.resolve(handler)
            members.append(
                GroupMemberView(
                    module=dep.target.module,
                    module_name=self._module_name(dep.target.module),
                    handler=handler,
                    axis=rule.axis if rule is not None else infer_axis(handler),
                    local_idxs=tuple(int(idx) for idx in idxs),
                    root_idxs=tuple(group[i].root_idxs),
                    measurable=rule is not None,
                    reason_unmeasurable=None if rule is not None else "no rule registered",
                )
            )

        common_kwargs = dict(
            root_module=root_module,
            root_module_name=self._module_name(root_module),
            root_handler=root_handler,
            atomic_size=atomic_size,
            channel_groups=channel_groups,
            members=tuple(members),
            attention_modules=attention_modules,
        )

        group_views: list[GroupView] = []
        if is_attention:
            group_views.extend(
                self._build_attention_group_views(
                    atomic_size=atomic_size,
                    channel_groups=channel_groups,
                    common_kwargs=common_kwargs,
                )
            )
        if len(group_views) == 0:
            prune_units = build_grouped_prune_units(atomic_size, channel_groups)
            group_views.append(
                GroupView(
                    group_id=f"{common_kwargs['root_module_name']}:{root_handler.__name__}",
                    axis=root_axis,
                    size=len(prune_units),
                    prune_units=tuple(prune_units),
                    **common_kwargs,
                )
            )
        return tuple(group_views)

    def _build_attention_group_views(
        self,
        atomic_size: int,
        channel_groups: int,
        common_kwargs: dict,
    ) -> list[GroupView]:
        group_views: list[GroupView] = []
        root_handler = common_kwargs["root_handler"]
        root_module_name = common_kwargs["root_module_name"]
        attention_modules: Sequence[nn.Module] = common_kwargs["attention_modules"]

        if self.config.prune_head_dims:
            prune_units = build_grouped_prune_units(atomic_size, channel_groups)
            group_views.append(
                GroupView(
                    group_id=f"{root_module_name}:{root_handler.__name__}:head_dim",
                    axis=StructureAxis.HEAD_DIM,
                    size=len(prune_units),
                    prune_units=tuple(prune_units),
                    **common_kwargs,
                )
            )

        if self.config.prune_num_heads and len(attention_modules) > 0:
            num_heads = max(self.num_heads[module] for module in attention_modules)
            prune_units = build_head_prune_units(atomic_size, num_heads)
            group_views.append(
                GroupView(
                    group_id=f"{root_module_name}:{root_handler.__name__}:head",
                    axis=StructureAxis.HEAD,
                    size=len(prune_units),
                    prune_units=tuple(prune_units),
                    **common_kwargs,
                )
            )

        return group_views

    def _detect_channel_groups(self) -> None:
        if len(self.num_heads) > 0:
            self.out_channel_groups.update(self.num_heads)
        for module in self.model.modules():
            layer_pruner = self.DG.get_pruner_of_module(module)
            if layer_pruner is None:
                continue
            in_ch_group = layer_pruner.get_in_channel_groups(module)
            out_ch_group = layer_pruner.get_out_channel_groups(module)
            if isinstance(module, ops.TORCH_CONV) and module.groups == module.out_channels:
                continue
            if in_ch_group > 1:
                self.in_channel_groups[module] = in_ch_group
            if out_ch_group > 1:
                self.out_channel_groups[module] = out_ch_group

    def _parameter_pruning_dim(self, parameter: nn.Parameter) -> int:
        if parameter in self._parameter_pruning_dims:
            return self._parameter_pruning_dims[parameter]
        return self.DG.module2node[parameter].pruning_dim

    def _module_name(self, module) -> str:
        if module in self.DG._module2name and self.DG._module2name[module] != "":
            return self.DG._module2name[module]
        if module in self.DG._param_to_name:
            return self.DG._param_to_name[module]
        return type(module).__name__

    def _ignored_layers(self) -> list[object]:
        return list(self.config.ignored_layers)

    def _is_atten_group(self, group) -> tuple[bool, list[nn.Module]]:
        is_attention = False
        qkv_layers = []
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            if self.DG.is_out_channel_pruning_fn(pruning_fn) and module in self.num_heads:
                qkv_layers.append(module)
                is_attention = True
        return is_attention, qkv_layers

    def _get_channel_groups(self, group) -> int:
        channel_groups = []
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            group_map = self.out_channel_groups if self.DG.is_out_channel_pruning_fn(pruning_fn) else self.in_channel_groups
            if module in group_map:
                channel_groups.append(group_map[module])
        if len(channel_groups) == 0:
            return 1
        return max(channel_groups)

    def _downstream_node_as_root_if_attention(self, group):
        is_attention = False
        downstream_dep = None
        idxs = None
        for dep, dep_idxs in group:
            if dep.source.module in self.num_heads and self.DG.is_out_channel_pruning_fn(dep.handler):
                is_attention = True
            if isinstance(dep.target.module, tuple(self.config.root_module_types)) and self.DG.is_in_channel_pruning_fn(dep.handler):
                downstream_dep = dep
                idxs = dep_idxs
        if is_attention and downstream_dep is not None:
            return self.DG.get_pruning_group(
                downstream_dep.target.module,
                downstream_dep.handler,
                idxs,
            )
        return None

    def _prunable_parameters(self) -> list[nn.Parameter]:
        parameters: dict[int, nn.Parameter] = {}
        for module in self.DG.module2node.keys():
            if isinstance(module, nn.Parameter):
                parameters[id(module)] = module
                continue
            if ops.module2type(module) not in self.DG.REGISTERED_PRUNERS:
                continue
            for parameter in module.parameters(recurse=False):
                if isinstance(parameter, nn.Parameter):
                    parameters[id(parameter)] = parameter
        return list(parameters.values())

    def _aggregate_by_layer(
        self,
        group_stats: Iterable[GroupSparsityStats],
    ) -> dict[str, LayerSparsityStats]:
        layer_accumulators = {}
        layer_modules = {}
        for group_stat in group_stats:
            module_name = group_stat.root_module_name
            if module_name not in layer_accumulators:
                layer_accumulators[module_name] = defaultdict(int)
                layer_modules[module_name] = group_stat.root_module

            key_prefix = group_stat.axis.value
            acc = layer_accumulators[module_name]
            acc[f"{key_prefix}_total"] += group_stat.stats.total
            acc[f"{key_prefix}_removed"] += group_stat.stats.removed
            acc[f"{key_prefix}_total_params"] += group_stat.stats.total_params
            acc[f"{key_prefix}_removed_params"] += group_stat.stats.removed_params

        by_layer = {}
        for module_name, acc in layer_accumulators.items():
            by_layer[module_name] = LayerSparsityStats(
                module=layer_modules[module_name],
                module_name=module_name,
                out_stats=self._stats_from_accumulator(acc, StructureAxis.OUT),
                in_stats=self._stats_from_accumulator(acc, StructureAxis.IN),
                param_stats=self._stats_from_accumulator(acc, StructureAxis.PARAM),
                head_stats=self._stats_from_accumulator(acc, StructureAxis.HEAD),
                head_dim_stats=self._stats_from_accumulator(acc, StructureAxis.HEAD_DIM),
            )
        return by_layer

    def _aggregate_global(
        self,
        group_stats: Iterable[GroupSparsityStats],
        total_params: int | None = None,
        removed_params: int | None = None,
    ) -> AxisSparsityStats:
        total = 0
        removed = 0
        summed_total_params = 0
        summed_removed_params = 0
        for group_stat in group_stats:
            total += group_stat.stats.total
            removed += group_stat.stats.removed
            summed_total_params += group_stat.stats.total_params
            summed_removed_params += group_stat.stats.removed_params
        return self._make_axis_stats(
            total=total,
            removed=removed,
            total_params=summed_total_params if total_params is None else total_params,
            removed_params=summed_removed_params if removed_params is None else removed_params,
        )

    def _stats_from_accumulator(self, acc, axis: StructureAxis) -> AxisSparsityStats | None:
        prefix = axis.value
        total = acc.get(f"{prefix}_total", 0)
        if total == 0:
            return None
        return self._make_axis_stats(
            total=total,
            removed=acc.get(f"{prefix}_removed", 0),
            total_params=acc.get(f"{prefix}_total_params", 0),
            removed_params=acc.get(f"{prefix}_removed_params", 0),
        )

    def _make_axis_stats(
        self,
        total: int,
        removed: int,
        total_params: int = 0,
        removed_params: int = 0,
    ) -> AxisSparsityStats:
        active = max(total - removed, 0)
        active_params = max(total_params - removed_params, 0)
        sparsity = 0.0 if total == 0 else removed / total
        return AxisSparsityStats(
            total=total,
            removed=removed,
            active=active,
            sparsity=sparsity,
            total_params=total_params,
            removed_params=removed_params,
            active_params=active_params,
        )

    def _all_zero(self, tensors: tuple[torch.Tensor, ...], tol: float) -> bool:
        for tensor in tensors:
            if torch.gt(tensor.detach().abs(), tol).any().item():
                return False
        return True

    def _accumulate_parameter_masks(
        self,
        accumulator: dict[int, tuple[torch.Tensor, torch.Tensor]],
        parameter_masks: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> None:
        # Used to merge parameter tensors, from different sources. Multiple groups can make use of the same weight tensor, thus to do propery account, we must ensure to merge them.          

        for parameter, mask in parameter_masks:
            key = id(parameter)
            if key not in accumulator:
                accumulator[key] = (parameter, mask.clone())
                continue
            existing_parameter, existing_mask = accumulator[key]
            if existing_parameter.shape != parameter.shape:
                raise ValueError("Mismatched parameter mask shapes for the same parameter id")
            existing_mask |= mask

    def _count_parameter_masks(
        self,
        accumulator: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> int:
        return sum(int(mask.sum().item()) for _, mask in accumulator.values())

    def _count_parameter_masks_by_parameter(
        self,
        parameter_masks: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> int:
        accumulator: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._accumulate_parameter_masks(accumulator, parameter_masks)
        return self._count_parameter_masks(accumulator)

    def _format_indices(self, idxs: Sequence[int]) -> str:
        if len(idxs) == 0:
            return "-"
        ordered = sorted(set(int(idx) for idx in idxs))
        ranges: list[str] = []
        start = ordered[0]
        end = start
        for idx in ordered[1:]:
            if idx == end + 1:
                end = idx
                continue
            ranges.append(self._format_index_range(start, end))
            start = end = idx
        ranges.append(self._format_index_range(start, end))
        return ", ".join(ranges)

    def _format_index_range(self, start: int, end: int) -> str:
        if start == end:
            return str(start)
        if end == start + 1:
            return f"{start},{end}"
        return f"[{start}:{end + 1})"
