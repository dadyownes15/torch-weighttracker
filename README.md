# torch-weighttracker

Tools for tracking structured weight sparsity, regularization signals, and
bit-operation estimates in PyTorch models.

The package builds a structural view of a model, compiles tensorized reduction
plans over that structure, and reuses those plans for training-time metrics and
regularizers.

```python
import torch
from torch import nn

from torch_weighttracker import WeightTracker

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
tracker = WeightTracker(model, example_inputs=torch.randn(1, 4))

print(tracker.view_structures())
```

## TRANSFORMS NOT FULLY SUPPORTED YET

## Installation

```bash
python -m pip install torch-weighttracker
```

Structured BOPs MAC accounting uses `fvcore` for baseline per-module MACs:

```bash
python -m pip install "torch-weighttracker[structured-bops]"
```

## Why Use It?

PyTorch makes it easy to inspect one parameter tensor at a time. Structured
compression often needs a different view:

- A channel can be coupled across convolutions, batch norms, linear layers, and
  residual paths.
- A transformer unit can mean an attention head, a head dimension, or a fused
  QKV slice rather than a simple row or column.
- A metric such as "active BOPs" depends on sparsity, module shape, MAC counts,
  and bitrates at the same time.
- A regularizer such as group lasso should penalize the coupled structural unit,
  not each weight tensor independently.

`WeightTracker` turns those coupled structures into canonical units, then lets
calculations operate over the canonical units with reusable tensor programs.

## Use Cases

Current use cases:

- Add structured group lasso to a training loss.
- Track active structured BOPs and compression rate during structured pruning,
  sparsity-aware training, or quantization-aware training (QAT).
- Inspect which modules participate in each channel, feature, head, or head-dim
  group.
- Build structural metrics that aggregate many weight tensors into one value per
  pruning unit.


## Group Lasso

Structured group lasso regularizes coupled units together. Layers can be
excluded per regularizer:

```python
from torch_weighttracker.regularizers import RegularizerType

group_lasso = tracker.create_regularizer(
    RegularizerType.GROUP_LASSO,
    ignore=[model.classifier],
)

loss = task_loss + 1e-4 * group_lasso()
loss.backward()
```

## Structured BOPs

Structured BOPs reports compression against a dense 32-bit baseline by default:

```python
import torch

from torch_weighttracker.trackers import TrackerType

metrics = tracker.create_tracker(
    TrackerType.STRUCTURED_BOPS,
    include=[model.layer3, model.layer4],
    ignore=[torch.nn.BatchNorm2d],
).track()

print(metrics["structured_bops_compression"])

raw_metrics = tracker.create_tracker(
    TrackerType.STRUCTURED_BOPS,
    include=[model.layer3, model.layer4],
    ignore=[torch.nn.BatchNorm2d],
    log_total_bops=True,
    log_layerwise_stats=True,
).track()

print(raw_metrics["structured_bops"])
print(raw_metrics["structured_bops_pr_module"])
print(raw_metrics["structured_bops_compression_rate_pr_module"])
```

`create_tracker` accepts a single `TrackerType`/string or a list of tracker
types/strings:

```python
tracker.create_tracker([TrackerType.STRUCTURED_BOPS, "unstructured_sparsity"])
metrics = tracker.track()
```

#### Formulation of the Structured BOPs Metric

For each weighted module $m$, WeightTracker multiplies the active structured MAC
count by that module's activation and weight bit widths [1]:

$$
\operatorname{StructuredBOPs}_m =
\operatorname{ActiveMACs}_m
\cdot
b^{\mathrm{act}}_m
\cdot
b^{\mathrm{weight}}_m
$$

The active MAC count scales the dense module MAC count by the active fraction of
each structural cost axis:

$$
\operatorname{ActiveMACs}_m =
\operatorname{BaselineMACs}_m
\cdot
\prod_{a \in A_m}
\frac{n^{\mathrm{active}}_{m,a}}{n^{\mathrm{baseline}}_{m,a}}
$$

Compression is reported against a dense 32-bit activation and 32-bit weight
baseline:

$$
\operatorname{BaselineBOPs}_m =
\operatorname{BaselineMACs}_m
\cdot
32
\cdot
32
$$

$$
\operatorname{CompressionRate} =
1 -
\frac{\sum_m \operatorname{StructuredBOPs}_m}
{\sum_m \operatorname{BaselineBOPs}_m}
$$

Where:

- $\operatorname{StructuredBOPs}_m$: active bit operations for weighted module
  $m$.
- $\operatorname{ActiveMACs}_m$: active MAC count after structured units are
  masked or pruned.
- $\operatorname{BaselineMACs}_m$: dense MAC count for module $m$ before
  structured pruning.
- $A_m$: structural cost axes for module $m$, such as input and output channel
  axes.
- $n^{\mathrm{active}}_{m,a}$: active size of cost axis $a$ for module $m$.
- $n^{\mathrm{baseline}}_{m,a}$: dense baseline size of cost axis $a$ for
  module $m$.
- $b^{\mathrm{act}}_m$: activation bit width for module $m$.
- $b^{\mathrm{weight}}_m$: weight bit width for module $m$.

### Comparison with Direct Removal and FLOP Count

For some model architectures, the BOPs calculation may differ from values
reported by other libraries. These differences mainly come from which layers and
operations are included. WeightTracker does not count elementwise operations
such as ReLU activations or bias terms.

The repository includes sanity notebooks comparing `fvcore.FlopCountAnalysis`
on physically pruned models with WeightTracker on fake-pruned models, where
weights are zeroed to match the equivalent hard-pruned structure.

Local sanity notebooks compare WeightTracker MAC accounting with physically
pruned models from Torch-Pruning. These dependencies are optional and are not
installed with the base package:

```bash
python -m pip install -e ".[dev-local]"
```

Then start Jupyter from the repository root and open the notebooks in
`sanity_checks/`.

## Unstructured Sparsity

Unstructured sparsity reports exact zero-weight fractions. The total is weighted
by each layer's number of weight elements, not averaged across layer fractions:

```python
import torch

from torch_weighttracker.trackers import TrackerType

metrics = tracker.create_tracker(
    TrackerType.UNSTRUCTURED_SPARSITY,
    include=[model.layer3, model.layer4],
    ignore=[torch.nn.BatchNorm2d],
).track()

print(metrics["unstructured_sparsity"])
print(metrics["layers"])
```

Values are fractions in `[0, 1]`. Parametrized fake quantization is measured
through the effective `module.weight`, so quantized zeros count as sparse
weights.

## NVIDIA 2:4 Sparsity

NVIDIA 2:4 sparsity reports block eligibility for supported weighted layers.
Linear and `MultiheadAttention` projection weights are grouped in contiguous
blocks of four along the input axis. Convolution weights shaped `[K, C, ...]`
are grouped along `C` for each output/spatial position.

```python
import torch

from torch_weighttracker.trackers import TrackerType

metrics = tracker.create_tracker(
    TrackerType.NVIDIA_2_4_SPARSITY,
    include=[model.layer3, model.layer4],
    ignore=[torch.nn.BatchNorm2d],
    log_layerwise_stats=True,
).track()

print(metrics["nvidia_2_4_sparsity/strict_block_fraction"])
print(metrics["nvidia_2_4_sparsity/nvidia_eligible_block_fraction"])
print(metrics["nvidia_2_4_sparsity/tail_elements"])
```

The strict fraction counts complete 4-value blocks with exactly two zeros. The
NVIDIA-eligible fraction counts blocks with at least two zeros, matching the
TensorRT eligibility rule. Tail elements are reported separately and prevent a
layer from counting as strict or eligible.

## Architecture

The main API is `WeightTracker`. Internally it is split into a few layers:

1. Dependency discovery: `WeightTracker` builds dependency groups from the model
   and `example_inputs`, or accepts precomputed groups.
2. Canonical units: `canonical_units.py` normalizes raw dependency groups into
   `CanonicalUnitGroup` objects. These give channels, features, attention heads,
   and head dimensions a shared unit index.
3. Reduction plans: `reductions/` and `plans/` compile module and unit mappings
   into segment and index operations that use PyTorch's efficient tensor
   computations.
4. Calculations: `calculations/` defines named calculation specs such as
   per-unit L2 norm, active units, parameters per unit, active MACs, and bitrates.
   Calculations can depend on each other and cache constant results.
5. Consumers: `regularizers/` and `trackers/` request the calculations they need,
   optionally with `include` and `ignore` contexts for selecting modules in a
   specific metric or regularizer.

The result is a small public surface with a reusable internal graph:

```text
model + example inputs
        |
        v
dependency groups -> canonical units -> reduction plans -> calculations
                                                          |
                                                          v
                                               regularizers and trackers
```

## Speed

Compared with a naive implementation, the current implementation gives the
following speedups on ResNet 20 on a RTX 3060:

| Comparison | Speedup | Naive extra allocation | WeightTracker extra allocation |
|---|---:|---:|---:|
| Group lasso | 15.421x | 197.0MiB | 197.0MiB |
| Structured BOPs | 2.582x | 1.7GiB | 195.9MiB |

## Status

This package is pre-1.0. Public APIs may still change while the tracker,
calculation, and regularizer surfaces settle.
 
## Future Work

1. Streamline definitions and method names across the codebase.
2. Improve calculation caching so repeated computations are not performed twice.
3. Improve compilation of computation plans for bigger speedups.
4. Improve memory management within calculations.
5. Write more comprehensive docstrings.

Future custom use cases will need a broader top-level `WeightTracker` API for
custom operations, custom layers, and generic group definitions.

## License

MIT

## References

[1] Wang et al., *Differentiable Joint Pruning and Quantization for Hardware Efficiency*, 2020.
