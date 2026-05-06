import torch.nn as nn
import torch

from torch_structracker.utils import initialize_from_groups

class ParamUnitSum(nn.Module):
    def __init__(self, groups, device=None, dtype=None) -> None:
        super().__init__()

        reductions, unit_count = initialize_from_groups(groups)

        self.unit_count = unit_count
        self.reducers = nn.ModuleList()
        self._dst_names: list[str] = []

        first_reducer = next(iter(reductions.values()))[0]
        first_weight = first_reducer.parameter_extractor.get()

        device = first_weight.device if device is None else device
        dtype = first_weight.dtype if dtype is None else dtype

        for i, (reducer, mapping) in enumerate(reductions.values()):
            self.reducers.append(reducer)

            if len(mapping) == 0:
                raise ValueError(f"Empty mapping for reducer {i}")

            self.register_buffer(
                f"dst_{i}",
                torch.tensor(mapping, dtype=torch.long, device=device),
                persistent=False,
            )

            self._dst_names.append(f"dst_{i}")

        self.register_buffer(
            "accumulator",
            torch.zeros(unit_count, device=device, dtype=dtype),
            persistent=False,
        )

        self.compile_indices()

    def compile_indices(self):
        self.destination_indices = tuple(
            getattr(self, name) for name in self._dst_names
        )
        return self

    @torch.no_grad()
    def forward(self):
        acc = self.accumulator
        acc.zero_()

        for reducer, dst in zip(self.reducers, self.destination_indices):
            acc.index_add_(0, dst, reducer())

        return acc