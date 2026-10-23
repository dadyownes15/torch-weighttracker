"""BOPs follow bit-width changes between track() calls.

BOPs per layer = MACs x input-activation bits x weight bits.

Run from the repo root: python -m sanity_checks.bitwidth_recheck
"""

import torch
import torch.nn as nn

from torch_weighttracker import WeightTracker


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.fc(x.mean((2, 3)))


# Layer-wise bit-widths, as a mixed-precision quantizer would hold them.
weight_bits = {"conv1": 8, "conv2": 8, "fc": 8}
output_bits = {"conv1": 8, "conv2": 8}


def weight_bitwidth_fn(name, module):
    return weight_bits[name]


def activation_bitwidth_fn(name, module):
    # Bit-width of the activations going INTO this layer.
    if name == "conv1":
        return 8  # the input image
    if name == "conv2":
        return output_bits["conv1"]  # conv2 consumes conv1's output
    if name == "fc":
        return output_bits["conv2"]  # fc consumes conv2's output


model = Net()
tracker = WeightTracker(
    model,
    torch.randn(1, 3, 16, 16),
    weight_bitwidth_fn=weight_bitwidth_fn,
    activation_bitwidth_fn=activation_bitwidth_fn,
)
bops = tracker.create_tracker(
    "structured_bops", log_layerwise_stats=True, log_total_bops=True
)


def report(title):
    modules = bops.track()["structured_bops"]["modules"]
    print(title)
    for layer in tracker.view_bitwidths().layers:
        layer_bops = modules[layer.name]["bops"]
        print(
            f"  {layer.name:<5} input {layer.activation_bits}-bit, "
            f"weights {layer.weight_bits}-bit: {layer_bops:>12,.0f} BOPs"
        )


report("All layers 8-bit:")

# During training, the quantizer lowers two bit-widths.
weight_bits["conv1"] = 4  # conv1's weights: conv1 BOPs halve
output_bits["conv1"] = 4  # conv1's outputs are conv2's inputs: conv2 BOPs halve

report("\nconv1 weights and conv1 outputs at 4-bit:")
