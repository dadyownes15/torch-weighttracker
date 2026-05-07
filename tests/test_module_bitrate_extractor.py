import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from torch_structracker import ModuleBitrateExtractor, StructureTracker
from torch_structracker.calculations import BitRatePrModule, CalculationType


class TinyWeightedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 1, bias=False)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class FakeQuantizer(nn.Module):
    def __init__(self, bitwidth):
        super().__init__()
        self.bitwidth = bitwidth

    def forward(self, weight):
        return weight

    def get_bitwidth(self):
        return self.bitwidth


class FakeQuantParametrization(nn.Module):
    def __init__(self, bitwidth):
        super().__init__()
        self.quantizer = FakeQuantizer(bitwidth)

    def forward(self, weight):
        return self.quantizer(weight)


def test_extractor_reads_codeq_style_weight_parametrization():
    model = TinyWeightedModel()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    parametrize.register_parametrization(
        model.fc1,
        "weight",
        FakeQuantParametrization(torch.tensor(3.0)),
    )

    extractor = ModuleBitrateExtractor(model)

    assert extractor.module_names == ("fc1", "fc2")
    torch.testing.assert_close(
        extractor.extract(),
        torch.tensor(
            [
                [8.0, 3.0],
                [32.0, 32.0],
            ]
        ),
    )


def test_extractor_reads_module_attribute_fallbacks():
    model = TinyWeightedModel()
    model.fc1.activation_bitrate = torch.tensor(4.0)
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 6

    extractor = ModuleBitrateExtractor(model)

    torch.testing.assert_close(
        extractor.extract(),
        torch.tensor(
            [
                [4.0, 2.0],
                [6.0, 6.0],
            ]
        ),
    )


def test_extractor_rejects_non_scalar_bitwidths():
    model = TinyWeightedModel()
    parametrize.register_parametrization(
        model.fc1,
        "weight",
        FakeQuantParametrization(torch.tensor([2.0, 3.0])),
    )

    extractor = ModuleBitrateExtractor(model)

    with pytest.raises(ValueError, match="weight bitrate must be scalar"):
        extractor.extract()


def test_extractor_rejects_multiple_weight_bitwidth_providers():
    model = TinyWeightedModel()
    parametrize.register_parametrization(
        model.fc1,
        "weight",
        FakeQuantParametrization(torch.tensor(2.0)),
    )
    parametrize.register_parametrization(
        model.fc1,
        "weight",
        FakeQuantParametrization(torch.tensor(3.0)),
    )

    extractor = ModuleBitrateExtractor(model)

    with pytest.raises(ValueError, match="multiple weight bitwidth providers"):
        extractor.extract()


def test_structure_tracker_creates_bitrate_calculation_without_groups():
    model = TinyWeightedModel()
    model.fc1.weight_bitrate = 2
    model.fc1.activation_bitrate = 4

    calculation = StructureTracker(model=model).get_calculation(
        CalculationType.BITRATE_PR_MODULE
    )

    assert isinstance(calculation, BitRatePrModule)
    assert calculation.module_names == ("fc1", "fc2")
    torch.testing.assert_close(
        calculation(),
        torch.tensor(
            [
                [4.0, 2.0],
                [32.0, 32.0],
            ]
        ),
    )
