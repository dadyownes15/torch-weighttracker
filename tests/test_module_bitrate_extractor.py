import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from torch_weighttracker.extractors.codeq_bitrate_extractor import (
    ModuleBitrateExtractor,
)


class TinyWeightedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 1, bias=False)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class TinyAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
        self.proj = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        y, _ = self.attn(x, x, x, need_weights=False)
        return self.proj(y)


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


def _bound_values(model: nn.Module) -> torch.Tensor:
    extractor = ModuleBitrateExtractor()
    refs = [
        extractor.bind(module)
        for _, module in ModuleBitrateExtractor.weighted_modules(model)
    ]
    assert all(ref is not None for ref in refs)
    return torch.stack([ref.get() for ref in refs if ref is not None])


def test_extractor_lists_weighted_modules():
    model = TinyWeightedModel()

    entries = tuple(ModuleBitrateExtractor.weighted_modules(model))

    assert tuple(name for name, _ in entries) == ("fc1", "fc2")


def test_extractor_lists_multihead_attention_parent_and_children():
    model = TinyAttentionModel()

    entries = tuple(ModuleBitrateExtractor.weighted_modules(model))

    assert tuple(name for name, _ in entries) == ("attn", "attn.out_proj", "proj")


def test_extractor_binds_multihead_attention_projection_bitrates():
    model = TinyAttentionModel()
    model.attn.activation_bitrate = 8
    model.attn.weight_bitrate = 4

    ref = ModuleBitrateExtractor().bind(model.attn)

    assert ref is not None
    assert ref.source_spec().shape == torch.Size([2])
    assert ref.source_spec().device == model.attn.in_proj_weight.device
    torch.testing.assert_close(ref.get(), torch.tensor([8.0, 4.0]))


def test_extractor_reads_codeq_style_weight_parametrization():
    model = TinyWeightedModel()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    parametrize.register_parametrization(
        model.fc1,
        "weight",
        FakeQuantParametrization(torch.tensor(3.0)),
    )

    torch.testing.assert_close(
        _bound_values(model),
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

    torch.testing.assert_close(
        _bound_values(model),
        torch.tensor(
            [
                [4.0, 2.0],
                [6.0, 6.0],
            ]
        ),
    )


def test_extractor_returns_none_for_unweighted_module():
    extractor = ModuleBitrateExtractor()

    assert extractor.bind(nn.ReLU()) is None


def test_extractor_ref_exposes_tensor_source_spec():
    model = TinyWeightedModel()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2

    ref = ModuleBitrateExtractor().bind(model.fc1)

    assert ref is not None
    assert ref.source_spec().shape == torch.Size([2])
    assert ref.source_spec().dtype == torch.float32
    assert ref.source_spec().device == model.fc1.weight.device
    torch.testing.assert_close(ref.get(), torch.tensor([8.0, 2.0]))


def test_extractor_rejects_non_scalar_bitwidths():
    model = TinyWeightedModel()
    parametrize.register_parametrization(
        model.fc1,
        "weight",
        FakeQuantParametrization(torch.tensor([2.0, 3.0])),
    )

    extractor = ModuleBitrateExtractor()

    with pytest.raises(ValueError, match="weight bitrate must be scalar"):
        extractor.bind(model.fc1)


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

    extractor = ModuleBitrateExtractor()

    with pytest.raises(ValueError, match="multiple weight bitwidth providers"):
        extractor.bind(model.fc1)
