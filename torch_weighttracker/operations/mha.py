import torch
import torch.nn as nn

from torch_weighttracker.extractors.extractor import SourceSpec, TensorSpec
from torch_weighttracker.operations.base import WeightOperation, WeightOperationType


def raise_mha_not_implemented(module: nn.MultiheadAttention):
    raise ValueError("MHA reducer plan creation is not implemented yet.")


def _reduce_rows(
    weight: torch.Tensor,
    operation_type: WeightOperationType | str,
) -> torch.Tensor:
    operation = WeightOperationType(operation_type)

    if weight.ndim == 0:
        raise ValueError("QKV operations require at least one source dimension.")

    flat = weight.reshape(weight.shape[0], -1)

    if operation == WeightOperationType.SUM:
        return flat.sum(dim=1)

    if operation == WeightOperationType.SQUARED_SUM:
        return flat.square().sum(dim=1)

    if operation == WeightOperationType.MEAN:
        return flat.mean(dim=1)

    if operation == WeightOperationType.COUNT:
        return torch.ones_like(flat).sum(dim=1)

    if operation == WeightOperationType.ACTIVE:
        return flat.ne(0).any(dim=1).to(dtype=weight.dtype)

    if operation == WeightOperationType.L1:
        return flat.abs().sum(dim=1)

    if operation == WeightOperationType.L2:
        return torch.linalg.vector_norm(flat, ord=2, dim=1)

    raise ValueError(f"Unsupported QKV operation: {operation_type}")


class QKVSourceOperation(WeightOperation):
    """Compatibility operation that returns reduced native Q/K/V source rows."""

    def __init__(self, operation_type: WeightOperationType | str):
        super().__init__()
        self.operation_type = WeightOperationType(operation_type)

    def forward(
        self,
        value: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(value, tuple):
            if len(value) != 3:
                raise ValueError("QKVSourceOperation requires exactly 3 tensors.")
            return torch.cat(
                tuple(_reduce_rows(weight, self.operation_type) for weight in value)
            )

        return _reduce_rows(value, self.operation_type)

    def identity_key(self):
        return ("qkv_source", self.operation_type.value)

    def output_spec(self, source_spec: SourceSpec) -> TensorSpec:
        if isinstance(source_spec, TensorSpec):
            if len(source_spec.shape) == 0:
                raise ValueError("QKV source tensor must have at least one dimension.")
            output_length = int(source_spec.shape[0])
            return TensorSpec(
                shape=torch.Size([output_length]),
                dtype=source_spec.dtype,
                device=source_spec.device,
            )

        if len(source_spec) != 3:
            raise ValueError("QKVSourceOperation requires exactly 3 source specs.")

        dtype, device = _common_tuple_dtype_device(source_spec)
        output_length = sum(int(spec.shape[0]) for spec in source_spec)
        return TensorSpec(
            shape=torch.Size([output_length]),
            dtype=dtype,
            device=device,
        )


class QKVSemanticOperation(WeightOperation):
    """Return the final semantic channel/head/head-dim vector for QKV weights."""

    def __init__(
        self,
        operation_type: WeightOperationType | str,
        embed_dim: int,
        num_heads: int | None = None,
        mode: str = "channel",
    ):
        super().__init__()
        self.operation_type = WeightOperationType(operation_type)
        self.embed_dim = int(embed_dim)
        self.num_heads = None if num_heads is None else int(num_heads)
        self.mode = str(mode)

        if self.embed_dim <= 0:
            raise ValueError("QKVSemanticOperation.embed_dim must be positive.")

        if self.mode not in {"channel", "head", "head_dim"}:
            raise ValueError(f"Unknown QKV semantic mode: {self.mode}")

        if self.mode != "channel":
            if self.num_heads is None or self.num_heads <= 0:
                raise ValueError(
                    "QKVSemanticOperation.num_heads must be positive for "
                    f"{self.mode!r} mode."
                )

            if self.embed_dim % self.num_heads != 0:
                raise ValueError("QKV embed_dim must be divisible by num_heads.")

    def forward(
        self,
        value: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        values = self._semantic_row_values(value)

        if self.mode == "channel":
            return values.sum(dim=0)

        assert self.num_heads is not None
        head_dim = self.embed_dim // self.num_heads
        values = values.reshape(3, self.num_heads, head_dim)

        if self.mode == "head":
            return values.sum(dim=(0, 2))

        if self.mode == "head_dim":
            return values.sum(dim=(0, 1))

        raise ValueError(f"Unknown QKV semantic mode: {self.mode}")

    def _semantic_row_values(
        self,
        value: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(value, tuple):
            if len(value) != 3:
                raise ValueError("QKVSemanticOperation requires exactly 3 tensors.")

            row_values = tuple(
                _reduce_rows(weight, self.operation_type) for weight in value
            )

            for values in row_values:
                if values.numel() != self.embed_dim:
                    raise ValueError(
                        "Separate QKV tensors must each have embed_dim source rows."
                    )

            return torch.stack(row_values, dim=0)

        expected_rows = 3 * self.embed_dim
        if value.shape[0] != expected_rows:
            raise ValueError(
                "Fused QKV tensor has "
                f"{value.shape[0]} source rows, expected {expected_rows}."
            )

        return _reduce_rows(value, self.operation_type).reshape(3, self.embed_dim)

    def identity_key(self):
        return (
            "qkv_semantic",
            self.operation_type.value,
            self.embed_dim,
            self.num_heads,
            self.mode,
        )

    def output_spec(self, source_spec: SourceSpec) -> TensorSpec:
        dtype, device = self._validate_source_spec(source_spec)

        if self.mode == "channel":
            output_length = self.embed_dim
        elif self.mode == "head":
            assert self.num_heads is not None
            output_length = self.num_heads
        elif self.mode == "head_dim":
            assert self.num_heads is not None
            output_length = self.embed_dim // self.num_heads
        else:
            raise ValueError(f"Unknown QKV semantic mode: {self.mode}")

        return TensorSpec(
            shape=torch.Size([output_length]),
            dtype=dtype,
            device=device,
        )

    def _validate_source_spec(
        self,
        source_spec: SourceSpec,
    ) -> tuple[torch.dtype, torch.device]:
        if isinstance(source_spec, TensorSpec):
            expected_rows = 3 * self.embed_dim
            if (
                len(source_spec.shape) == 0
                or int(source_spec.shape[0]) != expected_rows
            ):
                raise ValueError(
                    "Fused QKV source has "
                    f"{_source_rows(source_spec)} "
                    f"rows, expected {expected_rows}."
                )
            return source_spec.dtype, source_spec.device

        if len(source_spec) != 3:
            raise ValueError("QKVSemanticOperation requires exactly 3 source specs.")

        for spec in source_spec:
            if len(spec.shape) == 0 or int(spec.shape[0]) != self.embed_dim:
                raise ValueError(
                    "Separate QKV sources must each have embed_dim source rows."
                )

        return _common_tuple_dtype_device(source_spec)


class MultiheadAttentionSemanticOperation(WeightOperation):
    """Return semantic channel/head/head-dim values for all MHA projection weights."""

    def __init__(
        self,
        operation_type: WeightOperationType | str,
        embed_dim: int,
        num_heads: int | None = None,
        mode: str = "channel",
    ):
        super().__init__()
        self.operation_type = WeightOperationType(operation_type)
        self.embed_dim = int(embed_dim)
        self.num_heads = None if num_heads is None else int(num_heads)
        self.mode = str(mode)

        if self.embed_dim <= 0:
            raise ValueError(
                "MultiheadAttentionSemanticOperation.embed_dim must be positive."
            )

        if self.mode not in {"channel", "head", "head_dim"}:
            raise ValueError(f"Unknown MHA semantic mode: {self.mode}")

        if self.mode != "channel":
            if self.num_heads is None or self.num_heads <= 0:
                raise ValueError(
                    "MultiheadAttentionSemanticOperation.num_heads must be positive "
                    f"for {self.mode!r} mode."
                )
            if self.embed_dim % self.num_heads != 0:
                raise ValueError("MHA embed_dim must be divisible by num_heads.")

    def forward(
        self,
        value: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return torch.stack(
            tuple(
                self._reduce_unit(value, indices)
                for indices in self._unit_embed_indices()
            )
        )

    def _reduce_unit(
        self,
        value: torch.Tensor | tuple[torch.Tensor, ...],
        embed_indices: tuple[int, ...],
    ) -> torch.Tensor:
        tensors: list[torch.Tensor] = []

        if isinstance(value, tuple):
            if len(value) == 2:
                in_proj_weight, out_proj_weight = value
                tensors.append(_select_fused_mha_unit(in_proj_weight, embed_indices))
                tensors.append(_select_square_unit(out_proj_weight, embed_indices))
                return _reduce_flat_tensors(tensors, self.operation_type)

            if len(value) == 4:
                q_weight, k_weight, v_weight, out_proj_weight = value
                tensors.extend(
                    _select_separate_mha_unit(weight, embed_indices, self.embed_dim)
                    for weight in (q_weight, k_weight, v_weight)
                )
                tensors.append(_select_square_unit(out_proj_weight, embed_indices))
                return _reduce_flat_tensors(tensors, self.operation_type)

            if len(value) == 3:
                tensors.extend(
                    _select_separate_mha_unit(weight, embed_indices, self.embed_dim)
                    for weight in value
                )
                return _reduce_flat_tensors(tensors, self.operation_type)

            raise ValueError(
                "MultiheadAttentionSemanticOperation requires a fused "
                "(in_proj, out_proj) tuple or separate (q, k, v, out_proj) tuple."
            )

        tensors.append(_select_fused_mha_unit(value, embed_indices))
        return _reduce_flat_tensors(tensors, self.operation_type)

    def _unit_embed_indices(self) -> tuple[tuple[int, ...], ...]:
        if self.mode == "channel":
            return tuple((index,) for index in range(self.embed_dim))

        assert self.num_heads is not None
        head_dim = self.embed_dim // self.num_heads

        if self.mode == "head":
            return tuple(
                tuple(range(head * head_dim, (head + 1) * head_dim))
                for head in range(self.num_heads)
            )

        if self.mode == "head_dim":
            return tuple(
                tuple(head * head_dim + dim for head in range(self.num_heads))
                for dim in range(head_dim)
            )

        raise ValueError(f"Unknown MHA semantic mode: {self.mode}")

    def identity_key(self):
        return (
            "mha_semantic",
            self.operation_type.value,
            self.embed_dim,
            self.num_heads,
            self.mode,
        )

    def output_spec(self, source_spec: SourceSpec) -> TensorSpec:
        dtype, device = self._validate_source_spec(source_spec)

        if self.mode == "channel":
            output_length = self.embed_dim
        elif self.mode == "head":
            assert self.num_heads is not None
            output_length = self.num_heads
        elif self.mode == "head_dim":
            assert self.num_heads is not None
            output_length = self.embed_dim // self.num_heads
        else:
            raise ValueError(f"Unknown MHA semantic mode: {self.mode}")

        return TensorSpec(
            shape=torch.Size([output_length]),
            dtype=dtype,
            device=device,
        )

    def _validate_source_spec(
        self,
        source_spec: SourceSpec,
    ) -> tuple[torch.dtype, torch.device]:
        if isinstance(source_spec, TensorSpec):
            _validate_fused_qkv_spec(source_spec, self.embed_dim)
            return source_spec.dtype, source_spec.device

        if len(source_spec) == 2:
            in_proj_spec, out_proj_spec = source_spec
            _validate_fused_qkv_spec(in_proj_spec, self.embed_dim)
            _validate_square_spec(out_proj_spec, self.embed_dim)
            return _common_tuple_dtype_device(source_spec)

        if len(source_spec) == 4:
            for spec in source_spec[:3]:
                _validate_separate_qkv_spec(spec, self.embed_dim)
            _validate_square_spec(source_spec[3], self.embed_dim)
            return _common_tuple_dtype_device(source_spec)

        if len(source_spec) == 3:
            for spec in source_spec:
                _validate_separate_qkv_spec(spec, self.embed_dim)
            return _common_tuple_dtype_device(source_spec)

        raise ValueError(
            "MultiheadAttentionSemanticOperation requires a fused "
            "(in_proj, out_proj) source or separate (q, k, v, out_proj) source."
        )


class FusedQKVEmbedDimOperation(QKVSemanticOperation):
    def __init__(self, operation_type: WeightOperationType | str, embed_dim: int):
        super().__init__(
            operation_type=operation_type,
            embed_dim=embed_dim,
            mode="channel",
        )


class FusedQKVHeadOperation(QKVSemanticOperation):
    def __init__(
        self,
        operation_type: WeightOperationType | str,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__(
            operation_type=operation_type,
            embed_dim=int(num_heads) * int(head_dim),
            num_heads=num_heads,
            mode="head",
        )


class FusedQKVHeadDimOperation(QKVSemanticOperation):
    def __init__(
        self,
        operation_type: WeightOperationType | str,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__(
            operation_type=operation_type,
            embed_dim=int(num_heads) * int(head_dim),
            num_heads=num_heads,
            mode="head_dim",
        )


class SeparateQKVHeadOperation(FusedQKVHeadOperation):
    pass


def _select_fused_mha_unit(
    weight: torch.Tensor,
    embed_indices: tuple[int, ...],
) -> torch.Tensor:
    embed_dim = weight.shape[0] // 3
    expected_rows = 3 * embed_dim
    if weight.ndim < 2 or weight.shape[0] != expected_rows:
        raise ValueError("Fused MHA in_proj_weight must have shape [3E, E].")

    rows = torch.zeros(weight.shape[0], device=weight.device, dtype=torch.bool)
    cols = torch.zeros(weight.shape[1], device=weight.device, dtype=torch.bool)
    index = torch.tensor(embed_indices, device=weight.device, dtype=torch.long)

    for offset in (0, embed_dim, 2 * embed_dim):
        rows[index + offset] = True

    if weight.shape[1] == embed_dim:
        cols[index] = True

    return weight[rows[:, None] | cols[None, :]]


def _select_separate_mha_unit(
    weight: torch.Tensor,
    embed_indices: tuple[int, ...],
    embed_dim: int,
) -> torch.Tensor:
    if weight.ndim < 2 or weight.shape[0] != embed_dim:
        raise ValueError("Separate MHA projection weights must have shape [E, *].")

    rows = torch.zeros(weight.shape[0], device=weight.device, dtype=torch.bool)
    cols = torch.zeros(weight.shape[1], device=weight.device, dtype=torch.bool)
    index = torch.tensor(embed_indices, device=weight.device, dtype=torch.long)
    rows[index] = True

    if weight.shape[1] == embed_dim:
        cols[index] = True

    return weight[rows[:, None] | cols[None, :]]


def _select_square_unit(
    weight: torch.Tensor,
    embed_indices: tuple[int, ...],
) -> torch.Tensor:
    if weight.ndim < 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError("MHA out_proj weight must be a square matrix.")

    rows = torch.zeros(weight.shape[0], device=weight.device, dtype=torch.bool)
    cols = torch.zeros(weight.shape[1], device=weight.device, dtype=torch.bool)
    index = torch.tensor(embed_indices, device=weight.device, dtype=torch.long)
    rows[index] = True
    cols[index] = True
    return weight[rows[:, None] | cols[None, :]]


def _reduce_flat_tensors(
    tensors: list[torch.Tensor],
    operation_type: WeightOperationType,
) -> torch.Tensor:
    if len(tensors) == 0:
        raise ValueError("MHA semantic reduction requires at least one tensor.")

    flat = torch.cat(tuple(tensor.reshape(-1) for tensor in tensors))

    if operation_type == WeightOperationType.SUM:
        return flat.sum()

    if operation_type == WeightOperationType.SQUARED_SUM:
        return flat.square().sum()

    if operation_type == WeightOperationType.MEAN:
        return flat.mean()

    if operation_type == WeightOperationType.COUNT:
        return flat.new_tensor(float(flat.numel()))

    if operation_type == WeightOperationType.ACTIVE:
        return flat.ne(0).any().to(dtype=flat.dtype)

    if operation_type == WeightOperationType.L1:
        return flat.abs().sum()

    if operation_type == WeightOperationType.L2:
        return torch.linalg.vector_norm(flat, ord=2)

    raise ValueError(f"Unsupported MHA operation: {operation_type}")


def _validate_fused_qkv_spec(source_spec: TensorSpec, embed_dim: int) -> None:
    expected_rows = 3 * embed_dim
    if len(source_spec.shape) < 2 or int(source_spec.shape[0]) != expected_rows:
        raise ValueError(
            "Fused MHA source has "
            f"{_source_rows(source_spec)} "
            f"rows, expected {expected_rows}."
        )


def _validate_separate_qkv_spec(source_spec: TensorSpec, embed_dim: int) -> None:
    if len(source_spec.shape) < 2 or int(source_spec.shape[0]) != embed_dim:
        raise ValueError(
            "Separate MHA projection sources must each have embed_dim source rows."
        )


def _validate_square_spec(source_spec: TensorSpec, embed_dim: int) -> None:
    if (
        len(source_spec.shape) < 2
        or int(source_spec.shape[0]) != embed_dim
        or int(source_spec.shape[1]) != embed_dim
    ):
        raise ValueError("MHA out_proj source must have shape [embed_dim, embed_dim].")


def _source_rows(source_spec: TensorSpec) -> int:
    return 0 if len(source_spec.shape) == 0 else int(source_spec.shape[0])


def _common_tuple_dtype_device(
    source_spec: tuple[TensorSpec, ...],
) -> tuple[torch.dtype, torch.device]:
    if len(source_spec) == 0:
        raise ValueError("Tuple source spec must not be empty.")

    dtype = source_spec[0].dtype
    device = source_spec[0].device

    for spec in source_spec:
        if spec.dtype != dtype:
            raise ValueError("QKV source tensors must have matching dtype.")
        if spec.device != device:
            raise ValueError("QKV source tensors must live on the same device.")

    return dtype, device
