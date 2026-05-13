from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import Enum

import torch
from torch import nn

from torch_structracker.calculations import CalcType


class TrackerType(str, Enum):
    STRUCTURED_BOPS = "structured_bops"


class BaseTracker(nn.Module, ABC):
    required_calculations: tuple[CalcType, ...] = ()

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


def tracker_class_for_type(tracker_type: TrackerType | str):
    tracker_type = TrackerType(tracker_type)

    if tracker_type == TrackerType.STRUCTURED_BOPS:
        from torch_structracker.trackers.structured_bops import StructuredBOPs

        return StructuredBOPs

    raise ValueError(f"Unknown tracker type: {tracker_type.value}")
