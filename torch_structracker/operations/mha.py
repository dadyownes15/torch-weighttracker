import torch
import torch.nn as nn

from torch_structracker.operations.base import WeightOperation, WeightOperationType


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

    if operation == WeightOperationType.MEAN:
        return flat.mean(dim=1)

    if operation == WeightOperationType.COUNT:
        return torch.ones_like(flat).sum(dim=1)

    if operation == WeightOperationType.L1:
        return flat.abs().sum(dim=1)

    if operation == WeightOperationType.L2:
        return torch.sqrt((flat**2).sum(dim=1))

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
