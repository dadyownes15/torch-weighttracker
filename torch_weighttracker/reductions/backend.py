from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum


class ReductionBackend(str, Enum):
    """How compiled reduction plans execute.

    SPARSE: lower supported reductions into a few sparse matrix-vector products
        over one flattened source vector. Unsupported ops fall back to LOOP.
    LOOP: run one reduction and one scatter per op (the reference path).
    """

    SPARSE = "sparse"
    LOOP = "loop"


_CURRENT_BACKEND: ContextVar[ReductionBackend] = ContextVar(
    "torch_weighttracker_reduction_backend",
    default=ReductionBackend.SPARSE,
)


def get_reduction_backend() -> ReductionBackend:
    return _CURRENT_BACKEND.get()


@contextmanager
def use_reduction_backend(
    backend: ReductionBackend | str,
) -> Iterator[ReductionBackend]:
    """Select the backend for reduction calculations constructed in this scope."""
    token = _CURRENT_BACKEND.set(ReductionBackend(backend))
    try:
        yield _CURRENT_BACKEND.get()
    finally:
        _CURRENT_BACKEND.reset(token)
