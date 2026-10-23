"""BOPs follow bit-width changes between track() calls.

BOPs per layer = MACs x input-activation bits x weight bits.
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from torch_weighttracker import WeightTracker

model = nn.Sequential(
    OrderedDict(
        conv1=nn.Conv2d(3, 8, 3, padding=1),
        relu1=nn.ReLU(),
        conv2=nn.Conv2d(8, 8, 3, padding=1),
        relu2=nn.ReLU(),
        pool=nn.AdaptiveAvgPool2d(1),
        flatten=nn.Flatten(),
        fc=nn.Linear(8, 10),
    )
)

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


tracker = WeightTracker(
    model,
    torch.randn(1, 3, 16, 16),
    weight_bitwidth_fn=weight_bitwidth_fn,
    activation_bitwidth_fn=activation_bitwidth_fn,
)
bops = tracker.create_tracker(
    "structured_bops", log_total_bops=True, log_layerwise_stats=True
)

print(bops.track()["structured_bops"]["modules"])
# conv1: 3,538,944   conv2: 9,437,184   fc: 5,120

weight_bits["conv1"] = 4  # conv1's weights: conv1 BOPs halve
output_bits["conv1"] = 4  # conv1's outputs are conv2's inputs: conv2 BOPs halve

print(bops.track()["structured_bops"]["modules"])
# conv1: 1,769,472   conv2: 4,718,592   fc: 5,120
