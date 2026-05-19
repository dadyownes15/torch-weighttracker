from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from enum import Enum

import torch
from torch import nn

from torch_weighttracker.calculations import CalcType, CalculationContext
from torch_weighttracker.consumer_ignore import FilterItem


class TrackerType(str, Enum):
    STRUCTURED_BOPS = "structured_bops"
    L2_NORM_DISTRIBUTION = "l2_norm_distribution"
    UNSTRUCTURED_SPARSITY = "unstructured_sparsity"
    NVIDIA_2_4_SPARSITY = "nvidia_2_4_sparsity"
    GROUP_PRUNING_SUMMARY = "group_pruning_summary"


TrackerTypeInput = TrackerType | str
TrackerTypeSpec = TrackerTypeInput | Iterable[TrackerTypeInput]


def valid_tracker_type_values() -> tuple[str, ...]:
    return tuple(tracker_type.value for tracker_type in TrackerType)


def _valid_tracker_type_message() -> str:
    return ", ".join(valid_tracker_type_values())


def normalize_tracker_type(tracker_type: TrackerTypeInput) -> TrackerType:
    try:
        return TrackerType(tracker_type)
    except ValueError as exc:
        raise ValueError(
            f"{tracker_type!r} is not a valid TrackerType. "
            f"Available trackers: {_valid_tracker_type_message()}."
        ) from exc


def is_tracker_type_collection(tracker_type: TrackerTypeSpec) -> bool:
    return not isinstance(tracker_type, (str, TrackerType)) and isinstance(
        tracker_type, Iterable
    )


def normalize_tracker_types(tracker_type: TrackerTypeSpec) -> tuple[TrackerType, ...]:
    if is_tracker_type_collection(tracker_type):
        return tuple(normalize_tracker_type(item) for item in tracker_type)
    return (normalize_tracker_type(tracker_type),)


class BaseTracker(nn.Module, ABC):
    required_calculations: tuple[CalcType, ...] = ()

    @classmethod
    def calculation_context(
        cls,
        owner,
        *,
        include: Iterable[FilterItem] = (),
        ignore: Iterable[FilterItem] = (),
        **kwargs,
    ) -> CalculationContext | None:
        return None

    @classmethod
    def constructor_kwargs(
        cls,
        owner,
        *,
        context: CalculationContext | None = None,
        **kwargs,
    ) -> dict:
        return kwargs

    def __init__(
        self,
        calculations: Mapping[CalcType, nn.Module] | None = None,
    ) -> None:
        super().__init__()
        calculations = {} if calculations is None else calculations

        missing = [
            calc_type
            for calc_type in self.required_calculations
            if calc_type not in calculations
        ]

        if missing:
            raise ValueError(
                f"{self.__class__.__name__} is missing required calculations: {missing}"
            )

        self.calculations = nn.ModuleDict(
            {calc_type.name: module for calc_type, module in calculations.items()}
        )

    @abstractmethod
    def compute(
        self,
    ):
        raise NotImplementedError

    @abstractmethod
    def toMetric(self, result):
        raise NotImplementedError

    def track(self):
        with torch.no_grad():
            return self.toMetric(self.compute())

    def calc(self, calc_type: CalcType | str) -> nn.Module:
        calc_type = CalcType(calc_type)
        return self.calculations[calc_type.name]

    def compute_calculation(
        self,
        calc_type: CalcType | str,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self.calc(calc_type)(*args, **kwargs)


def tracker_class_for_type(tracker_type: TrackerTypeInput):
    tracker_type = normalize_tracker_type(tracker_type)

    if tracker_type == TrackerType.STRUCTURED_BOPS:
        from torch_weighttracker.trackers.structured_bops import StructuredBOPs

        return StructuredBOPs

    if tracker_type == TrackerType.L2_NORM_DISTRIBUTION:
        from torch_weighttracker.trackers.l2_norm_distribution import (
            L2NormDistribution,
        )

        return L2NormDistribution

    if tracker_type == TrackerType.UNSTRUCTURED_SPARSITY:
        from torch_weighttracker.trackers.unstructured_sparsity import (
            UnstructuredSparsity,
        )

        return UnstructuredSparsity

    if tracker_type == TrackerType.NVIDIA_2_4_SPARSITY:
        from torch_weighttracker.trackers.nvidia_2_4_sparsity import (
            Nvidia24Sparsity,
        )

        return Nvidia24Sparsity

    if tracker_type == TrackerType.GROUP_PRUNING_SUMMARY:
        from torch_weighttracker.trackers.group_pruning_summary import (
            GroupPruningSummary,
        )

        return GroupPruningSummary

    raise ValueError(f"Unknown tracker type: {tracker_type.value}")
