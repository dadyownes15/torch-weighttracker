import torch.nn as nn

from torch_structracker.calculations import (
    CalcType,
    create_calculation,
)

from torch_structracker.canonical_units import CanonicalUnitGroup, canonicalize_groups
from torch_structracker.operations import WeightOperationType
from torch_structracker.plans.bitrate_plan import create_codeq_bitrates
from torch_structracker.plans.unit_weight_operation_plan import create_group_member_plan
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
        groups=None,
        example_inputs=None,
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

        if groups is not None:
            self.groups = list(groups)
        elif example_inputs is not None and root_module_types is not None:
            self.groups = self._build_groups(
                example_inputs=example_inputs,
                root_module_types=root_module_types,
                forward_fn=forward_fn,
                output_transform=output_transform,
                unwrapped_parameters=unwrapped_parameters,
                customized_pruners=customized_pruners,
            )
        elif example_inputs is not None or root_module_types is not None:
            raise ValueError(
                "StructureTracker requires both example_inputs and "
                "root_module_types to build dependency groups."
            )
        else:
            self.groups = []

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
        self.regularizers = []
        self.trackers = []

    def _build_groups(
        self,
        example_inputs,
        root_module_types,
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
                root_module_types=root_module_types,
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

    def get_calculation(self, calculation_type: CalcType):
        calculation_type = CalcType(calculation_type)

        if calculation_type not in self.calculations:
            self.calculations[calculation_type] = self._create_calculation(
                calculation_type
            )

        return self.calculations[calculation_type]

    def _create_calculation(self, type: CalcType):

        match type:
            case CalcType.UNITS_TO_MODULE:
                #  plan
            case 
                CalcType.L2_NORM_PR_UNIT,
                CalcType.UNITS_TO_GROUP,
                CalcType.UNIT_ACTIVE_MASK,
                CalcType.BASELINE_GROUP_SIZES,
                CalcType.GROUP_CHANGE_EFFECT,
                CalcType.GROUP_SIZES,
       



    def ensure_calculations(self, calculation_types):
        return {
            calculation_type: self.get_calculation(calculation_type)
            for calculation_type in calculation_types
        }

    def create_tracker(self, tracker_type: TrackerType):
        tracker_type = TrackerType(tracker_type)
        tracker_cls = tracker_class_for_type(tracker_type)
        calculations = self.ensure_calculations(tracker_cls.required_calculations)
        tracker = tracker_cls(calculations=calculations)
        self.trackers.append(tracker)
        return tracker

    def create_regularizer(self, regularizer_type: RegularizerType):
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
        if len(self.groups) == 0:
            raise ValueError(
                f"{calculation_type.value} requires dependency groups. Pass groups "
                "when constructing StructTracker."
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
