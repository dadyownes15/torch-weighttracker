import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from torch_weighttracker import WeightTracker


class TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2, bias=False)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class FakeQuantizer(nn.Module):
    """Mimics a CoDeQ quantizer: get_bitwidth() returns a fresh CPU tensor."""

    def __init__(self, bits):
        super().__init__()
        self.register_buffer("bits", torch.tensor(float(bits)))

    def forward(self, weight):
        return weight

    def get_bitwidth(self):
        return self.bits.detach().clone().cpu()


class FakeQuantParametrization(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.quantizer = FakeQuantizer(bits)

    def forward(self, weight):
        return self.quantizer(weight)


def attach_quantizer(module: nn.Module, bits) -> FakeQuantizer:
    parametrization = FakeQuantParametrization(bits)
    parametrize.register_parametrization(module, "weight", parametrization)
    return parametrization.quantizer


def codeq_style_bitwidth_fn(activation_bits: dict[str, int]):
    def bitwidth_fn(name, module):
        weight_bits = 32
        if parametrize.is_parametrized(module, "weight"):
            for parametrization in module.parametrizations.weight:
                weight_bits = parametrization.quantizer.get_bitwidth()
        return activation_bits.get(name, 32), weight_bits

    return bitwidth_fn


def _module_bops(tracker, tracker_type: str) -> dict[str, float]:
    metrics = tracker.track()[tracker_type]
    return {name: values["bops"] for name, values in metrics["modules"].items()}


def _create_bops_tracker(tracker: WeightTracker, tracker_type: str):
    return tracker.create_tracker(
        tracker_type,
        log_total_bops=True,
        log_layerwise_stats=True,
    )


@pytest.mark.parametrize("backend", ["sparse", "loop"])
@pytest.mark.parametrize("tracker_type", ["structured_bops", "unstructured_bops"])
def test_bops_follow_bitwidth_changes_between_track_calls(
    tracker_type, backend
) -> None:
    torch.manual_seed(0)
    model = TwoLayer()
    quantizer = attach_quantizer(model.fc1, 8)
    activation_bits = {"fc2": 8}
    owner = WeightTracker(
        model,
        torch.randn(1, 4),
        reduction_backend=backend,
        bitwidth_fn=codeq_style_bitwidth_fn(activation_bits),
    )
    bops = _create_bops_tracker(owner, tracker_type)
    before = _module_bops(bops, tracker_type)

    with torch.no_grad():
        quantizer.bits.fill_(4)
    activation_bits["fc2"] = 6
    metrics = bops.track()[tracker_type]
    after = {name: values["bops"] for name, values in metrics["modules"].items()}

    assert after["fc1"] == pytest.approx(before["fc1"] * 4 / 8)
    assert after["fc2"] == pytest.approx(before["fc2"] * 6 / 8)
    assert metrics["bops"] == pytest.approx(after["fc1"] + after["fc2"])


def test_untouched_layers_keep_their_bops() -> None:
    model = TwoLayer()
    fc1 = attach_quantizer(model.fc1, 8)
    attach_quantizer(model.fc2, 8)
    owner = WeightTracker(
        model,
        torch.randn(1, 4),
        bitwidth_fn=codeq_style_bitwidth_fn({}),
    )
    bops = _create_bops_tracker(owner, "structured_bops")
    before = _module_bops(bops, "structured_bops")

    with torch.no_grad():
        fc1.bits.fill_(2)
    after = _module_bops(bops, "structured_bops")

    assert after["fc1"] == pytest.approx(before["fc1"] / 4)
    assert after["fc2"] == pytest.approx(before["fc2"])


def test_quantizer_attached_after_create_tracker_is_picked_up() -> None:
    model = TwoLayer()
    owner = WeightTracker(
        model,
        torch.randn(1, 4),
        bitwidth_fn=codeq_style_bitwidth_fn({}),
    )
    bops = _create_bops_tracker(owner, "structured_bops")
    before = _module_bops(bops, "structured_bops")

    attach_quantizer(model.fc1, 4)
    after = _module_bops(bops, "structured_bops")

    assert after["fc1"] == pytest.approx(before["fc1"] * 4 / 32)


def test_without_bitwidth_fn_every_layer_is_full_precision() -> None:
    model = TwoLayer()
    model.fc1.weight_bitrate = 4
    model.fc1.activation_bitrate = 4
    attach_quantizer(model.fc2, 4)
    owner = WeightTracker(model, torch.randn(1, 4))
    metrics = _create_bops_tracker(owner, "structured_bops").track()

    assert metrics["structured_bops"]["bops"] == pytest.approx(
        metrics["structured_bops"]["baseline"]
    )
    assert {
        layer.name: (layer.activation_bits, layer.weight_bits, layer.source)
        for layer in owner.view_bitwidths().layers
    } == {
        "fc1": (32, 32, "default"),
        "fc2": (32, 32, "default"),
    }


def test_bitwidth_fn_receives_bops_module_names() -> None:
    model = TwoLayer()
    seen = []

    def bitwidth_fn(name, module):
        seen.append((name, module))
        return 8, 8

    owner = WeightTracker(model, torch.randn(1, 4), bitwidth_fn=bitwidth_fn)
    metrics = _create_bops_tracker(owner, "structured_bops").track()

    names = set(metrics["structured_bops"]["modules"])
    assert names == {"fc1", "fc2"}
    assert {name for name, _ in seen} == names
    assert all(module is model.get_submodule(name) for name, module in seen)


def test_view_bitwidths_reflects_current_values() -> None:
    model = TwoLayer()
    quantizer = attach_quantizer(model.fc1, 8)
    owner = WeightTracker(
        model,
        torch.randn(1, 4),
        bitwidth_fn=codeq_style_bitwidth_fn({"fc2": 6}),
    )

    def view():
        return {
            layer.name: (layer.activation_bits, layer.weight_bits, layer.source)
            for layer in owner.view_bitwidths().layers
        }

    assert view() == {
        "fc1": (32, 8, "bitwidth_fn"),
        "fc2": (6, 32, "bitwidth_fn"),
    }
    with torch.no_grad():
        quantizer.bits.fill_(3)
    assert view()["fc1"] == (32, 3, "bitwidth_fn")
    assert isinstance(owner.view_bitwidths().layers[0].weight_bits, int)


@pytest.mark.parametrize(
    "bad_bits",
    [5.3, 0, -4, True, torch.tensor(2.5), float("nan")],
    ids=["fractional", "zero", "negative", "bool", "fractional-tensor", "nan"],
)
def test_invalid_bitwidth_values_raise_with_layer_name(bad_bits) -> None:
    model = TwoLayer()
    bits = {"fc1": 8}

    def bitwidth_fn(name, module):
        return 32, bits.get(name, 32)

    owner = WeightTracker(model, torch.randn(1, 4), bitwidth_fn=bitwidth_fn)
    bops = _create_bops_tracker(owner, "structured_bops")
    bops.track()

    bits["fc1"] = bad_bits
    with pytest.raises(ValueError, match=r"'fc1'.*weight"):
        bops.track()
    with pytest.raises(ValueError, match=r"'fc1'.*weight"):
        owner.view_bitwidths()


def test_invalid_bitwidth_at_creation_raises() -> None:
    model = TwoLayer()
    owner = WeightTracker(
        model,
        torch.randn(1, 4),
        bitwidth_fn=lambda name, module: (4.5, 8),
    )

    with pytest.raises(ValueError, match=r"activation"):
        _create_bops_tracker(owner, "structured_bops").track()


@pytest.mark.parametrize(
    "result",
    [8, (8,), (8, 8, 8), "88", None],
    ids=["int", "1-tuple", "3-tuple", "str", "none"],
)
def test_bitwidth_fn_must_return_a_pair(result) -> None:
    owner = WeightTracker(
        TwoLayer(),
        torch.randn(1, 4),
        bitwidth_fn=lambda name, module: result,
    )

    with pytest.raises(TypeError, match=r"'fc1'"):
        owner.view_bitwidths()


def test_errors_inside_bitwidth_fn_name_the_layer() -> None:
    def bitwidth_fn(name, module):
        raise KeyError(name)

    owner = WeightTracker(TwoLayer(), torch.randn(1, 4), bitwidth_fn=bitwidth_fn)

    with pytest.raises(RuntimeError, match=r"'fc1'") as error:
        owner.view_bitwidths()
    assert isinstance(error.value.__cause__, KeyError)


def test_whole_number_float_and_tensor_bitwidths_are_accepted() -> None:
    owner = WeightTracker(
        TwoLayer(),
        torch.randn(1, 4),
        bitwidth_fn=lambda name, module: (8.0, torch.tensor(4.0)),
    )

    layers = owner.view_bitwidths().layers

    assert [(layer.activation_bits, layer.weight_bits) for layer in layers] == [
        (8, 4),
        (8, 4),
    ]


def test_bitwidth_fn_must_be_callable() -> None:
    with pytest.raises(TypeError, match="bitwidth_fn"):
        WeightTracker(TwoLayer(), torch.randn(1, 4), bitwidth_fn=8)
