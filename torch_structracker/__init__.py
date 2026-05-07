"""Public API for torch-structracker."""

try:
    from torch_structracker._version import __version__
except ModuleNotFoundError:
    __version__ = "0.0.0"

from torch_structracker.calculations import (
    BaseCalculation,
    BitRatePrModule,
    CalculationType,
    StructuredUnitSum,
)
from torch_structracker.bitrate_extractor import ModuleBitrateExtractor
from torch_structracker.regularizers import BaseRegularizer, RegularizerType
from torch_structracker.structure_tracker import StructureTracker
from torch_structracker.trackers import BaseTracker, TrackerType

__all__ = [
    "BaseCalculation",
    "BaseRegularizer",
    "BaseTracker",
    "BitRatePrModule",
    "CalculationType",
    "ModuleBitrateExtractor",
    "RegularizerType",
    "StructureTracker",
    "StructuredUnitSum",
    "TrackerType",
    "__version__",
]
