import torch

from torch_structracker.calculations.base import BaseCalculation, CalculationType


def initialize_axis_slices_and_updates_from_groups(groups, device=None):
    raise NotImplementedError(
        "StructuredUnitCount axis/update initialization is not implemented yet."
    )


class StructuredUnitCountFromNorm(BaseCalculation):
    calculation_type = CalculationType.STRUCTURED_UNIT_COUNT_FROM_NORM

    def __init__(
        self,
        groups,
        initial_counts,
        device=None,
        dtype=torch.float32,
        eps=0.0,
    ):
        super().__init__()

        self.eps = float(eps)

        axis_slices, updates, unit_count = initialize_axis_slices_and_updates_from_groups(
            groups,
            device=device,
        )

        initial_counts = torch.as_tensor(
            initial_counts,
            device=device,
            dtype=dtype,
        ).reshape(-1)

        if initial_counts.numel() != unit_count:
            raise ValueError(
                f"initial_counts has {initial_counts.numel()} entries, "
                f"but groups imply {unit_count} structure units."
            )

        self.unit_count = unit_count
        self.axis_slices = axis_slices
        self.updates = updates

        self.register_buffer("initial_counts", initial_counts, persistent=False)

        self.register_buffer(
            "zero_mask",
            torch.empty(unit_count, device=device, dtype=torch.bool),
            persistent=False,
        )

        self.register_buffer(
            "axis_zero_counts",
            torch.empty(len(axis_slices), device=device, dtype=dtype),
            persistent=False,
        )

        self.register_buffer(
            "accumulator",
            torch.empty(unit_count, device=device, dtype=dtype),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, unit_norms):
        if unit_norms.ndim != 1:
            unit_norms = unit_norms.reshape(-1)

        if unit_norms.numel() != self.unit_count:
            raise ValueError(
                f"Expected {self.unit_count} unit norms, got {unit_norms.numel()}"
            )

        torch.less_equal(unit_norms, self.eps, out=self.zero_mask)

        acc = self.accumulator
        acc.zero_()

        acc.add_(self.initial_counts * self.zero_mask.to(acc.dtype))

        az = self.axis_zero_counts
        az.zero_()

        for axis_id, axis_slice in enumerate(self.axis_slices):
            az[axis_id] = self.zero_mask[axis_slice.indices].sum().to(az.dtype)

        for update in self.updates:
            n_zero = az[update.source_axis_id]

            if n_zero.item() == 0:
                continue

            target_indices = self.axis_slices[update.target_axis_id].indices
            acc[target_indices].add_(update.delta * n_zero)

        torch.minimum(acc, self.initial_counts, out=acc)

        return acc
