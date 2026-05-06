import torch
import torch.nn as nn

from torch_structracker.operations.base import WeightOperation, WeightOperationType


def raise_mha_not_implemented(module: nn.MultiheadAttention):
    raise ValueError("MHA reducer plan creation is not implemented yet.")


class _MHAOperation(WeightOperation):
    def __init__(self, operation_type: WeightOperationType | str) -> None:
        super().__init__()
        self.operation_type = WeightOperationType(operation_type)
        self._row_reducer = _row_reducer_for_operation(self.operation_type)

    def identity_key(self):
        return (type(self), self.operation_type)

    def _reduce_rows(self, weight: torch.Tensor) -> torch.Tensor:
        return self._row_reducer(weight)


class QKVSourceOperation(_MHAOperation):
    """Reduce QKV source rows without collapsing their structural grouping."""

    def forward(self, qkv_source) -> torch.Tensor:
        if isinstance(qkv_source, torch.Tensor):
            return self._reduce_rows(qkv_source).reshape(-1)

        if len(qkv_source) != 3:
            raise ValueError("QKVSourceOperation expects fused QKV or q, k, v weights.")

        q_weight, k_weight, v_weight = qkv_source
        return torch.cat(
            [
                self._reduce_rows(q_weight).reshape(-1),
                self._reduce_rows(k_weight).reshape(-1),
                self._reduce_rows(v_weight).reshape(-1),
            ]
        )


class FusedQKVEmbedDimOperation(_MHAOperation):
    def __init__(
        self,
        operation_type: WeightOperationType | str,
        embed_dim: int,
    ) -> None:
        super().__init__(operation_type)
        self.embed_dim = int(embed_dim)

    def identity_key(self):
        return (*super().identity_key(), self.embed_dim)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        row_values = self._reduce_rows(weight)
        return row_values.reshape(3, self.embed_dim).sum(dim=0)


class FusedQKVHeadOperation(_MHAOperation):
    def __init__(
        self,
        operation_type: WeightOperationType | str,
        num_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__(operation_type)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)

    def identity_key(self):
        return (*super().identity_key(), self.num_heads, self.head_dim)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        row_values = self._reduce_rows(weight)
        return row_values.reshape(3, self.num_heads, self.head_dim).sum(dim=(0, 2))


class SeparateQKVHeadOperation(_MHAOperation):
    def __init__(
        self,
        operation_type: WeightOperationType | str,
        num_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__(operation_type)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)

    def identity_key(self):
        return (*super().identity_key(), self.num_heads, self.head_dim)

    def forward(self, weights) -> torch.Tensor:
        if len(weights) != 3:
            raise ValueError("SeparateQKVHeadOperation expects q, k, and v weights.")

        q_weight, k_weight, v_weight = weights
        row_values = torch.stack(
            [
                self._reduce_rows(q_weight),
                self._reduce_rows(k_weight),
                self._reduce_rows(v_weight),
            ],
        )
        return row_values.reshape(3, self.num_heads, self.head_dim).sum(dim=(0, 2))


def _row_reducer_for_operation(operation_type: WeightOperationType):
    if operation_type == WeightOperationType.SUM:
        return _sum_rows

    if operation_type == WeightOperationType.MEAN:
        return _mean_rows

    if operation_type == WeightOperationType.COUNT:
        return _count_rows

    if operation_type == WeightOperationType.L1:
        return _l1_rows

    if operation_type == WeightOperationType.L2:
        return _l2_rows

    raise ValueError(f"Unknown MHA operation type: {operation_type}")


def _sum_rows(weight: torch.Tensor) -> torch.Tensor:
    return weight.sum(dim=1)


def _mean_rows(weight: torch.Tensor) -> torch.Tensor:
    return weight.mean(dim=1)


def _count_rows(weight: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(weight).sum(dim=1)


def _l1_rows(weight: torch.Tensor) -> torch.Tensor:
    return weight.abs().sum(dim=1)


def _l2_rows(weight: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((weight**2).sum(dim=1))
