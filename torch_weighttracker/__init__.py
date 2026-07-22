from importlib.metadata import PackageNotFoundError, version

from torch_weighttracker.canonical_units import SeparateQKVAttentionSpec
from torch_weighttracker.weight_tracker import (
    FakePruneUnitResult,
    PruneUnitResult,
    WeightTracker,
)

__all__ = [
    "FakePruneUnitResult",
    "PruneUnitResult",
    "SeparateQKVAttentionSpec",
    "WeightTracker",
]

try:
    __version__ = version("torch-weighttracker")
except PackageNotFoundError:
    __version__ = "0+unknown"
