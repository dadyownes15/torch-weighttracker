from importlib.metadata import PackageNotFoundError, version

from torch_weighttracker.canonical_units import SeparateQKVAttentionSpec
from torch_weighttracker.reductions.backend import (
    ReductionBackend,
    use_reduction_backend,
)
from torch_weighttracker.reductions.sparse import SparseReductionFallbackWarning
from torch_weighttracker.weight_tracker import (
    FakePruneUnitResult,
    PruneUnitResult,
    WeightTracker,
)

__all__ = [
    "FakePruneUnitResult",
    "PruneUnitResult",
    "ReductionBackend",
    "SeparateQKVAttentionSpec",
    "SparseReductionFallbackWarning",
    "WeightTracker",
    "use_reduction_backend",
]

try:
    __version__ = version("torch-weighttracker")
except PackageNotFoundError:
    __version__ = "0+unknown"
