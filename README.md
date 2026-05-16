# torch-weighttracker

Package for tracking structured weight sparsity, regularization signals,
and bit-operation estimates in torch modules.

The API is centered on `WeightTracker`:

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
## Tensorized cross weight operations

Weighttracker builds an interface for doing cross weight tensor operations on models efficiently. Using the "Computation plans" and "Calculation" classes, we compile a set of torch modules which execute tensors and mapping operations on training device with the "minimal" set of repeated operations. 

## Use case

Weighttrackers primary use case for now is for calculating structural depedency based loss terms & metric evaluations, such as structured sparsity & structured compression rates, and group lasso. However, the code has been made such that it can be used for any weight traversering operations with some modifications.   



## Group lasso

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


## Speed
Comparing with a naive implementation we get the following speed ups: 

- Group lasso: 15.503x
  - Naive: 4.6540s total, 232.698ms/step
  - Weighttracker: 0.3002s total, 15.010ms/step
- Structured BOPs: 2.531x
  - Naive: 0.6757s total, 33.783ms/step
  - Weighttracker: 0.2669s total, 13.346ms/step

| Comparison | Speedup | Naive extra alloc | Weighttracker extra alloc |
|---|---:|---:|---:|
| Group lasso | 15.421x | 197.0MiB | 197.0MiB |
| Structured BOPs | 2.582x | 1.7GiB | 195.9MiB |


## Status

This package is pre-1.0. Public APIs may still change while the tracker,
calculation, and regularizer surfaces settle.

## Future work:

1. Streamlining defintions and methods across the code for a unified and more compressed and perhabs more understandable API.
2. Implement Calculation caching, such that computations are not computed twice
3. Improve compilations of computation plans
4. Improve memory management within calculations
5. Write more comprehensive docstrings

For future use cases, an update of the toplevel API `WeightTracker` is needed, including the ability to input custom operations, custom layers, generic group defintions etc.   

## License

MIT
