"""Public API for torch-structracker."""

from importlib.util import find_spec

from _version import __version__

__all__ = ["__version__"]

if find_spec(__name__ + ".tracker") is not None:
    from torch_structracker.tracker import StructTracker

    __all__.append("StructTracker")
