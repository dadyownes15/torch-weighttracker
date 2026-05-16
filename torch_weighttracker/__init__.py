from importlib.metadata import PackageNotFoundError, version

from torch_weighttracker.weight_tracker import WeightTracker

__all__ = ["WeightTracker"]

try:
    __version__ = version("torch-weighttracker")
except PackageNotFoundError:
    __version__ = "0+unknown"
