import torch
from torch_weighttracker.calculations.base import Calculation


class StaticCalc(Calculation):
    def __init__(
        self,
        value: torch.Tensor,
        *,
        persistent: bool = False,
    ) -> None:
        super().__init__()
        
        self.register_buffer(
            "value",
            value.detach().clone(),
            persistent=persistent,
        )

    def forward(self) -> torch.Tensor:
        return self.value
