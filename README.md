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

Structured BOPs compares active bit operations against a dense 32-bit baseline:

```python
import torch

from torch_weighttracker.trackers import TrackerType

metrics = tracker.create_tracker(
    TrackerType.STRUCTURED_BOPS,
    ignore=[torch.nn.BatchNorm2d],
    log_compression_rate=True,
).track()

print(metrics["structured_bops"])
print(metrics["structured_bops_baseline"])
print(metrics["structured_bops_compression_rate"])
```

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
   optionally with an `ignore` context for excluding modules from a specific
   metric or regularizer.

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
