"""Sparse lowering of reduction plans.

A reduction plan runs one small reduction plus one scatter per op. Most ops are
"reduce slices of a tensor with an elementwise function, then add the slice
values into output positions", which is linear in the elementwise values. This
module lowers those ops into sparse matrices over one flattened source vector:

    out = bias + M @ f(z)                 (no slice post-op)
    out = bias + D @ g(E @ f(z))          (slice post-op such as any() or sqrt)

where z concatenates every distinct source tensor, f is an elementwise function
(square, abs, != 0, ...), E maps elements to slices, g is the slice post-op and
D scatters slices into the output. Ops that cannot be expressed this way stay on
the loop path.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from torch_weighttracker.extractors.extractor import TensorSpec
from torch_weighttracker.operations import generic
from torch_weighttracker.reductions.backend import (
    ReductionBackend,
    get_reduction_backend,
)

SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "cuda"})
SUPPORTED_DTYPES = frozenset({torch.float32, torch.float64})

# Ops whose source tensor is larger than this stay on the loop path. Small ops
# are dominated by per-op launch overhead, which the sparse path removes; large
# ops already reduce efficiently with dense kernels, and lowering them costs more
# memory traffic (one index per element) than it saves.
MAX_LOWERED_SOURCE_ELEMENTS: dict[str, int] = {
    "cpu": 1 << 18,
    "cuda": 1 << 18,
}


class SparseReductionFallbackWarning(RuntimeWarning):
    """A reduction plan could not use the sparse backend and runs as a loop."""


@dataclass(frozen=True)
class SparseForm:
    """How one reduction op maps elements of its source onto output slices."""

    elementwise: str
    post: str = "none"
    dims: tuple[int, ...] | None = ()
    scale: float = 1.0
    shift: float = 0.0
    mean: bool = False


@dataclass(frozen=True)
class SparseKind:
    """One lowered group of entries sharing an elementwise and a slice post-op."""

    elementwise: str
    post: str
    matrices: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class PlanEntry:
    op: Any
    slice_sources: tuple[int, ...] | None
    destinations: tuple[int, ...]


_ELEMENTWISE: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "identity": lambda z: z,
    "square": torch.square,
    "abs": torch.abs,
    "nonzero": lambda z: z.ne(0).to(dtype=z.dtype),
    "ones": torch.ones_like,
}

_DIMENSIONAL_FORMS: dict[type, tuple[str, str]] = {
    generic.SumWeight: ("identity", "none"),
    generic.SquaredSumWeight: ("square", "none"),
    generic.MeanWeight: ("identity", "none"),
    generic.CountWeight: ("ones", "none"),
    generic.ActiveWeight: ("nonzero", "any"),
    generic.L1Weight: ("abs", "none"),
    generic.L2Weight: ("square", "sqrt"),
}

_ELEMENTWISE_FORMS: dict[type, str] = {
    generic.ElementwiseSumWeight: "identity",
    generic.ElementwiseSquaredSumWeight: "square",
    generic.ElementwiseL2Weight: "abs",
}


def sparse_form(reduction: Any, shape: torch.Size) -> SparseForm | None:
    """Describe a reduction as elementwise-then-slice-sum, or None if it is not."""
    reduction_type = type(reduction)

    if reduction_type in _ELEMENTWISE_FORMS:
        return SparseForm(elementwise=_ELEMENTWISE_FORMS[reduction_type], dims=())

    if reduction_type in _DIMENSIONAL_FORMS:
        dims = _normalize_dims(reduction.dim, rank=len(shape))
        if getattr(reduction, "keepdim", False) not in (True, False):
            return None
        elementwise, post = _DIMENSIONAL_FORMS[reduction_type]
        if dims == ():
            # Only ActiveWeight defines dim=() as elementwise; torch reduces all
            # dims for an empty dim tuple on the other reductions.
            if reduction_type is not generic.ActiveWeight:
                return None
            post = "none"
        return SparseForm(
            elementwise=elementwise,
            post=post,
            dims=dims,
            mean=reduction_type is generic.MeanWeight,
        )

    affine = getattr(reduction, "elementwise_affine", None)
    if callable(affine):
        scale, shift = affine()
        return SparseForm(
            elementwise="identity",
            dims=(),
            scale=float(scale),
            shift=float(shift),
        )

    return None


def plan_entries(plan) -> tuple[PlanEntry, ...]:
    """Normalize plan entries into (op, slice sources, destinations) triples.

    Order: segment entries, indexed entries, indexed gather entries.
    """
    entries: list[PlanEntry] = []
    for entry in plan.segment_entries:
        start, length = int(entry.start), int(entry.length)
        entries.append(PlanEntry(entry.op, None, tuple(range(start, start + length))))
    for entry in plan.indexed_entries:
        entries.append(PlanEntry(entry.op, None, tuple(entry.destination_indices)))
    for entry in plan.indexed_gather_entries:
        entries.append(
            PlanEntry(
                entry.op,
                tuple(entry.source_indices),
                tuple(entry.destination_indices),
            )
        )
    return tuple(entries)


def compile_sparse_plan(
    entries: Sequence[PlanEntry],
    *,
    output_spec: TensorSpec,
    input_spec: TensorSpec | None = None,
) -> tuple[SparseReductionExecutor | None, tuple[bool, ...]]:
    """Lower supported entries into a sparse executor.

    Args:
        entries: Normalized plan entries.
        output_spec: The plan output spec.
        input_spec: For pipeline plans, the spec of the forward input that every
            op reads. Reduction plans read their ops' source refs instead.

    Returns:
        The executor (None when nothing could be lowered) and one flag per entry
        that is True when the entry is handled by the executor.
    """
    dtype = output_spec.dtype
    device = torch.device(output_spec.device)
    output_length = int(output_spec.shape[0])
    if device.type not in SUPPORTED_DEVICE_TYPES or dtype not in SUPPORTED_DTYPES:
        return None, tuple(False for _ in entries)

    pipeline = input_spec is not None
    sources: dict[Any, tuple[Any, int]] = {}
    source_refs: list[Any] = []
    source_numels: list[int] = []
    total_elements = _numel(input_spec.shape) if pipeline else 0

    kinds: dict[tuple[str, str], _KindAccumulator] = {}
    bias = torch.zeros(output_length, dtype=torch.float64)
    has_bias = False
    flags: list[bool] = []

    for entry in entries:
        lowered = _lower_entry(entry, pipeline=pipeline, input_spec=input_spec)
        if lowered is None:
            flags.append(False)
            continue

        form, shape, source_ref = lowered
        spec_ok = True
        if not pipeline:
            spec = source_ref.source_spec()
            spec_ok = (
                spec.dtype == dtype
                and torch.device(spec.device) == device
                and _numel(shape) <= MAX_LOWERED_SOURCE_ELEMENTS[device.type]
            )
        if not spec_ok:
            flags.append(False)
            continue

        slice_of_element, num_slices = _slice_ids(shape, form.dims)
        if num_slices != int(entry.op.output_length):
            flags.append(False)
            continue

        if pipeline:
            offset = 0
        else:
            key = source_ref.identity_key()
            if key not in sources:
                sources[key] = (source_ref, total_elements)
                source_refs.append(source_ref)
                source_numels.append(_numel(shape))
                total_elements += _numel(shape)
            offset = sources[key][1]

        slice_sources = (
            torch.arange(num_slices)
            if entry.slice_sources is None
            else torch.tensor(entry.slice_sources, dtype=torch.long)
        )
        destinations = torch.tensor(entry.destinations, dtype=torch.long)
        if slice_sources.numel() != destinations.numel():
            flags.append(False)
            continue

        scale = form.scale
        if form.mean and num_slices > 0:
            scale = scale * num_slices / _numel(shape)

        accumulator = kinds.setdefault(
            (form.elementwise, form.post),
            _KindAccumulator(form.elementwise, form.post),
        )
        accumulator.add(
            slice_of_element=slice_of_element,
            num_slices=num_slices,
            offset=offset,
            slice_sources=slice_sources,
            destinations=destinations,
            scale=scale,
        )
        if form.shift != 0.0 and destinations.numel() > 0:
            bias.index_add_(
                0,
                destinations,
                torch.full((destinations.numel(),), form.shift, dtype=torch.float64),
            )
            has_bias = True
        flags.append(True)

    if not any(flags):
        return None, tuple(flags)

    executor = SparseReductionExecutor(
        output_length=output_length,
        num_elements=total_elements,
        dtype=dtype,
        device=device,
        kinds=tuple(kinds.values()),
        bias=bias if has_bias else None,
        source_refs=None if pipeline else tuple(source_refs),
        source_numels=None if pipeline else tuple(source_numels),
    )
    return executor, tuple(flags)


def select_backend(
    plan,
    *,
    input_spec: TensorSpec | None,
    loop_output: Callable[[tuple[bool, ...], torch.Tensor | None], torch.Tensor],
    label: str,
) -> tuple[SparseReductionExecutor | None, tuple[bool, ...]]:
    """Pick the execution path for a plan under the current backend.

    Returns the sparse executor (or None) and one flag per plan entry, True when
    the executor handles that entry. Lowered entries are checked once against
    the loop path; any failure falls back to the loop with a warning.
    """
    entries = plan_entries(plan)
    loop_everything = (None, tuple(False for _ in entries))
    if get_reduction_backend() != ReductionBackend.SPARSE or not entries:
        return loop_everything

    try:
        executor, flags = compile_sparse_plan(
            entries,
            output_spec=plan.output_spec,
            input_spec=input_spec,
        )
        if executor is None:
            return loop_everything

        probe = None
        if input_spec is not None:
            probe = torch.rand(
                tuple(input_spec.shape),
                dtype=input_spec.dtype,
                device=input_spec.device,
            )
        with torch.no_grad():
            expected = loop_output(flags, probe)
            actual = executor(probe)
        tolerance = (
            1e-5 * max(1.0, float(expected.abs().max())) if expected.numel() else 0
        )
        if not torch.allclose(actual, expected, rtol=1e-4, atol=tolerance):
            raise RuntimeError(
                "sparse result differs from the loop result by "
                f"{float((actual - expected).abs().max()):.3e}"
            )
    except Exception as error:  # noqa: BLE001 - any failure means "use the loop"
        warnings.warn(
            f"{label}: sparse reduction backend unavailable ({error}); "
            "using the loop backend.",
            SparseReductionFallbackWarning,
            stacklevel=3,
        )
        return loop_everything

    return executor, flags


class SparseReductionExecutor(nn.Module):
    """Evaluates lowered entries with sparse matrix-vector products."""

    def __init__(
        self,
        *,
        output_length: int,
        num_elements: int,
        dtype: torch.dtype,
        device: torch.device,
        kinds: Sequence[_KindAccumulator],
        bias: torch.Tensor | None,
        source_refs: tuple[Any, ...] | None,
        source_numels: tuple[int, ...] | None,
    ) -> None:
        super().__init__()
        self.output_length = int(output_length)
        self.num_elements = int(num_elements)
        self.source_refs = source_refs
        self.source_numels = source_numels
        self._kinds: list[tuple[str, str, tuple[_SparseMatrix, ...]]] = []
        self.nnz = 0

        for index, kind in enumerate(kinds):
            matrices = kind.finalize(
                output_length=self.output_length,
                num_elements=self.num_elements,
                dtype=dtype,
            )
            registered = []
            for name, matrix in zip(("a", "b"), matrices, strict=False):
                registered.append(
                    _SparseMatrix.register(
                        self, f"kind{index}_{name}", matrix, device=device
                    )
                )
                self.nnz += int(matrix.values().numel())
            self._kinds.append((kind.elementwise, kind.post, tuple(registered)))

        self.has_bias = bias is not None
        self.register_buffer(
            "bias",
            (
                torch.zeros(self.output_length, dtype=dtype, device=device)
                if bias is None
                else bias.to(dtype=dtype, device=device)
            ),
            persistent=False,
        )

    @property
    def kinds(self) -> tuple[tuple[str, str], ...]:
        return tuple((elementwise, post) for elementwise, post, _ in self._kinds)

    def matrices(self) -> tuple[SparseKind, ...]:
        """The lowered matrices, for inspection.

        Each kind computes ``out += matrices[0] @ f(z)`` when it has one matrix,
        or ``out += matrices[1] @ post(matrices[0] @ f(z))`` when it has two.
        Columns of the first matrix follow ``source_refs`` order.
        """
        return tuple(
            SparseKind(
                elementwise=elementwise,
                post=post,
                matrices=tuple(matrix.csr() for matrix in matrices),
            )
            for elementwise, post, matrices in self._kinds
        )

    def gather_sources(self) -> torch.Tensor:
        if self.source_refs is None or self.source_numels is None:
            raise RuntimeError("Pipeline executors read the forward input.")
        values = []
        for ref, numel in zip(self.source_refs, self.source_numels, strict=True):
            value = ref.get()
            if value.numel() != numel:
                raise RuntimeError(
                    "A reduction source changed size after the calculation was "
                    "built. Rebuild the calculation after physical pruning."
                )
            values.append(value.reshape(-1))
        if len(values) == 1:
            return values[0]
        return torch.cat(values)

    def forward(self, z: torch.Tensor | None = None) -> torch.Tensor:
        if z is None:
            z = self.gather_sources()
        else:
            z = z.reshape(-1)
            if z.numel() != self.num_elements:
                raise RuntimeError(
                    f"Expected {self.num_elements} input elements, got {z.numel()}."
                )

        out = self.bias.clone() if self.has_bias else None
        for elementwise, post, matrices in self._kinds:
            values = _ELEMENTWISE[elementwise](z)
            if post == "none":
                contribution = matrices[0].matvec(values)
            else:
                slices = matrices[0].matvec(values)
                contribution = matrices[1].matvec(_apply_post(post, slices))
            out = contribution if out is None else out + contribution
        if out is None:
            return self.bias.clone()
        return out


class _KindAccumulator:
    """Collects COO triplets for one (elementwise, post) kind."""

    def __init__(self, elementwise: str, post: str) -> None:
        self.elementwise = elementwise
        self.post = post
        self.rows: list[torch.Tensor] = []
        self.cols: list[torch.Tensor] = []
        self.vals: list[torch.Tensor] = []
        self.d_rows: list[torch.Tensor] = []
        self.d_cols: list[torch.Tensor] = []
        self.num_slices = 0

    def add(
        self,
        *,
        slice_of_element: torch.Tensor,
        num_slices: int,
        offset: int,
        slice_sources: torch.Tensor,
        destinations: torch.Tensor,
        scale: float,
    ) -> None:
        if destinations.numel() == 0 or slice_of_element.numel() == 0:
            return

        if self.post == "none":
            rows, elements = _expand_slices(
                slice_of_element, num_slices, slice_sources, destinations
            )
            self.rows.append(rows)
            self.cols.append(elements + offset)
            self.vals.append(torch.full((rows.numel(),), scale, dtype=torch.float64))
            return

        base = self.num_slices
        self.rows.append(slice_of_element + base)
        self.cols.append(torch.arange(slice_of_element.numel()) + offset)
        self.vals.append(
            torch.full((slice_of_element.numel(),), scale, dtype=torch.float64)
        )
        self.d_rows.append(destinations)
        self.d_cols.append(slice_sources + base)
        self.num_slices += num_slices

    def finalize(
        self,
        *,
        output_length: int,
        num_elements: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        if self.post == "none":
            return (
                _coo_to_csr(
                    self.rows,
                    self.cols,
                    self.vals,
                    (output_length, num_elements),
                    dtype,
                ),
            )

        ones = [torch.ones(rows.numel(), dtype=torch.float64) for rows in self.d_rows]
        return (
            _coo_to_csr(
                self.rows, self.cols, self.vals, (self.num_slices, num_elements), dtype
            ),
            _coo_to_csr(
                self.d_rows, self.d_cols, ones, (output_length, self.num_slices), dtype
            ),
        )


class _SparseMatrix:
    """CSR matrix stored as dense module buffers so `.to()` moves it."""

    def __init__(self, owner: nn.Module, prefix: str, size: tuple[int, int]) -> None:
        self.owner = owner
        self.prefix = prefix
        self.size = size
        self._csr_cache: tuple[tuple, torch.Tensor] | None = None
        self._transpose_cache: tuple[tuple, torch.Tensor] | None = None

    @classmethod
    def register(
        cls,
        owner: nn.Module,
        prefix: str,
        matrix: torch.Tensor,
        *,
        device: torch.device,
    ) -> _SparseMatrix:
        crow, col = _compact_indices(
            matrix.crow_indices(), matrix.col_indices(), int(matrix.shape[1])
        )
        owner.register_buffer(f"{prefix}_crow", crow.to(device), persistent=False)
        owner.register_buffer(f"{prefix}_col", col.to(device), persistent=False)
        owner.register_buffer(
            f"{prefix}_val", matrix.values().to(device), persistent=False
        )
        return cls(owner, prefix, (int(matrix.shape[0]), int(matrix.shape[1])))

    def _buffers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            getattr(self.owner, f"{self.prefix}_crow"),
            getattr(self.owner, f"{self.prefix}_col"),
            getattr(self.owner, f"{self.prefix}_val"),
        )

    def csr(self) -> torch.Tensor:
        crow, col, val = self._buffers()
        key = (crow.data_ptr(), col.data_ptr(), val.data_ptr(), val.device)
        if self._csr_cache is None or self._csr_cache[0] != key:
            self._csr_cache = (key, _csr_tensor(crow, col, val, self.size))
        return self._csr_cache[1]

    def transpose_csr(self) -> torch.Tensor:
        crow, _, val = self._buffers()
        key = (crow.data_ptr(), val.data_ptr(), val.device)
        if self._transpose_cache is None or self._transpose_cache[0] != key:
            with _quiet_sparse_warnings():
                transposed = self.csr().to_sparse_coo().t().coalesce().to_sparse_csr()
            t_crow, t_col = _compact_indices(
                transposed.crow_indices(), transposed.col_indices(), self.size[0]
            )
            self._transpose_cache = (
                key,
                _csr_tensor(
                    t_crow, t_col, transposed.values(), (self.size[1], self.size[0])
                ),
            )
        return self._transpose_cache[1]

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        return _SparseMatVec.apply(self, vector)


class _SparseMatVec(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix: _SparseMatrix, vector: torch.Tensor) -> torch.Tensor:
        ctx.matrix = matrix
        return matrix.csr() @ vector

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.needs_input_grad[1]:
            return None, None
        return None, ctx.matrix.transpose_csr() @ grad_output


def _apply_post(post: str, slices: torch.Tensor) -> torch.Tensor:
    if post == "any":
        return slices.gt(0).to(dtype=slices.dtype)
    if post == "sqrt":
        positive = slices.gt(0)
        safe = torch.where(positive, slices, torch.ones_like(slices))
        return torch.where(positive, safe.sqrt(), torch.zeros_like(slices))
    raise ValueError(f"Unknown slice post-op: {post}")


def _lower_entry(
    entry: PlanEntry,
    *,
    pipeline: bool,
    input_spec: TensorSpec | None,
) -> tuple[SparseForm, torch.Size, Any] | None:
    op = entry.op
    reduction = getattr(op, "reduction", None)
    source_ref = getattr(op, "source_ref", None)
    if reduction is None or source_ref is None:
        return None

    source_spec = getattr(op, "source_spec", None)
    if not isinstance(source_spec, TensorSpec):
        return None

    shape = torch.Size(source_spec.shape)
    if pipeline:
        assert input_spec is not None
        if _numel(shape) != _numel(input_spec.shape):
            return None
    elif not callable(getattr(source_ref, "identity_key", None)):
        return None

    form = sparse_form(reduction, shape)
    if form is None:
        return None
    return form, shape, source_ref


def _normalize_dims(dim, *, rank: int) -> tuple[int, ...] | None:
    if dim is None:
        return None
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    normalized = set()
    for item in dims:
        value = int(item)
        if value < 0:
            value += rank
        if value < 0 or value >= rank:
            raise ValueError(f"Reduction dim {item} is outside tensor rank {rank}.")
        normalized.add(value)
    return tuple(sorted(normalized))


def _slice_ids(
    shape: torch.Size,
    dims: tuple[int, ...] | None,
) -> tuple[torch.Tensor, int]:
    """Return the output slice index of every element (row-major order)."""
    numel = _numel(shape)
    if dims is None:
        return torch.zeros(numel, dtype=torch.long), 1
    if dims == ():
        return torch.arange(numel), numel

    kept = [axis for axis in range(len(shape)) if axis not in dims]
    num_slices = _numel(torch.Size([shape[axis] for axis in kept]))
    view_shape = [shape[axis] if axis in kept else 1 for axis in range(len(shape))]
    slice_ids = torch.arange(num_slices).reshape(view_shape).expand(shape)
    return slice_ids.reshape(-1), num_slices


def _expand_slices(
    slice_of_element: torch.Tensor,
    num_slices: int,
    slice_sources: torch.Tensor,
    destinations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand (slice -> destination) pairs into (destination, element) pairs."""
    order = torch.argsort(slice_of_element, stable=True)
    counts = torch.bincount(slice_of_element, minlength=num_slices)
    starts = torch.cumsum(counts, 0) - counts

    pair_counts = counts[slice_sources]
    rows = torch.repeat_interleave(destinations, pair_counts)
    pair_offsets = torch.cumsum(pair_counts, 0) - pair_counts
    within = torch.arange(int(pair_counts.sum())) - torch.repeat_interleave(
        pair_offsets, pair_counts
    )
    elements = order[
        torch.repeat_interleave(starts[slice_sources], pair_counts) + within
    ]
    return rows, elements


def _coo_to_csr(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    size: tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    if rows:
        indices = torch.stack([torch.cat(rows), torch.cat(cols)])
        values = torch.cat(vals)
    else:
        indices = torch.zeros((2, 0), dtype=torch.long)
        values = torch.zeros(0, dtype=torch.float64)
    with _quiet_sparse_warnings():
        coo = torch.sparse_coo_tensor(
            indices, values, size, check_invariants=False
        ).coalesce()
        return (
            torch.sparse_coo_tensor(
                coo.indices(), coo.values().to(dtype), size, check_invariants=False
            )
            .coalesce()
            .to_sparse_csr()
        )


def _csr_tensor(
    crow: torch.Tensor,
    col: torch.Tensor,
    val: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    with _quiet_sparse_warnings():
        return torch.sparse_csr_tensor(
            crow, col, val, size=size, check_invariants=False
        )


@contextmanager
def _quiet_sparse_warnings() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Ss]parse.*beta.*")
        warnings.filterwarnings("ignore", message=".*[Ss]parse invariant checks.*")
        yield


def _compact_indices(
    crow: torch.Tensor,
    col: torch.Tensor,
    num_cols: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    limit = torch.iinfo(torch.int32).max
    nnz = 0 if crow.numel() == 0 else int(crow[-1])
    if nnz < limit and num_cols < limit:
        return crow.to(torch.int32), col.to(torch.int32)
    return crow, col


def _numel(shape) -> int:
    total = 1
    for size in shape:
        total *= int(size)
    return total
