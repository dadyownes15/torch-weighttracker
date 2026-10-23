import torch
import torch.nn as nn

from torch_weighttracker.extractors.bitwidth_extractor import (
    ModuleBitwidthExtractor,
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


def test_extractor_lists_weighted_modules():
    model = TinyWeightedModel()

    entries = tuple(ModuleBitwidthExtractor.weighted_modules(model))

    assert tuple(name for name, _ in entries) == ("fc1", "fc2")


def test_extractor_lists_multihead_attention_parent_and_children():
    model = TinyAttentionModel()

    entries = tuple(ModuleBitwidthExtractor.weighted_modules(model))

    assert tuple(name for name, _ in entries) == ("attn", "attn.out_proj", "proj")


def test_extractor_binds_multihead_attention_projection_bitwidths():
    model = TinyAttentionModel()
    extractor = ModuleBitwidthExtractor(
        activation_bitwidth_fn=lambda name, module: 8,
        weight_bitwidth_fn=lambda name, module: 4,
        module_names={model.attn: "attn"},
    )

    ref = extractor.bind(model.attn)

    assert ref is not None
    assert ref.name == "attn"
    assert ref.source_spec().shape == torch.Size([2])
    assert ref.source_spec().device == model.attn.in_proj_weight.device
    torch.testing.assert_close(ref.get(), torch.tensor([8.0, 4.0]))


def test_extractor_defaults_to_full_precision():
    model = TinyWeightedModel()

    ref = ModuleBitwidthExtractor().bind(model.fc1)

    assert ref is not None
    assert ref.source_spec().dtype == torch.float32
    torch.testing.assert_close(ref.get(), torch.tensor([32.0, 32.0]))


def test_extractor_ref_rereads_bitwidths_on_every_get():
    model = TinyWeightedModel()
    bits = {"fc1": (8, 8)}
    extractor = ModuleBitwidthExtractor(
        activation_bitwidth_fn=lambda name, module: bits[name][0],
        weight_bitwidth_fn=lambda name, module: bits[name][1],
        module_names={model.fc1: "fc1"},
    )
    ref = extractor.bind(model.fc1)
    assert ref is not None

    bits["fc1"] = (6, 4)

    torch.testing.assert_close(ref.get(), torch.tensor([6.0, 4.0]))


def test_extractor_returns_none_for_unweighted_module():
    extractor = ModuleBitwidthExtractor()

    assert extractor.bind(nn.ReLU()) is None
