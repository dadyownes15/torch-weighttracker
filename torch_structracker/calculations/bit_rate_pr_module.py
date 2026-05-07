from torch import Tensor

from torch_structracker.bitrate_extractor import ModuleBitrateExtractor
from torch_structracker.calculations.base import BaseCalculation


class BitRatePrModule(BaseCalculation):
    @classmethod
    def from_model(cls, model, device=None, dtype=None, **kwargs):
        return cls(model, device=device, dtype=dtype, **kwargs)

    def __init__(
        self,
        model,
        *,
        activation_default: float = 32.0,
        weight_default: float = 32.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.extractor = ModuleBitrateExtractor(
            model,
            activation_default=activation_default,
            weight_default=weight_default,
            device=device,
            dtype=dtype,
        )

    @property
    def module_names(self) -> tuple[str, ...]:
        return self.extractor.module_names

    def forward(self) -> Tensor:
        return self.extractor.extract()
