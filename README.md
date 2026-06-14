# torch-weighttracker

Track, regularize, and prune structured units in PyTorch models.

`torch-weighttracker` gives you a model-level view of sparsity: channels,
features, attention heads, head dimensions, and fused QKV slices are grouped into
canonical units, so metrics, pruning, and regularizers operate on the structure
you would actually compress.

```python
import torch
import timm

from torch_weighttracker import WeightTracker
from torch_weighttracker.integrations.timm import infer_vit_num_heads

model = timm.create_model("vit_base_patch16_224", pretrained=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
example_inputs = torch.randn(1, 3, 224, 224)

tracker = WeightTracker(
    model,
    example_inputs=example_inputs,
    num_heads=infer_vit_num_heads(model),
    prune_num_heads=True,
)

tracker.create_tracker("structured_bops", log_total_bops=True)
group_lasso = tracker.create_regularizer("group_lasso")

for inputs, targets in dataloader:
    optimizer.zero_grad()

    outputs = model(inputs)
    task_loss = criterion(outputs, targets)
    loss = task_loss + 1e-4 * group_lasso()

    loss.backward()
    optimizer.step()

    metrics = tracker.track()

tracker.prune_zero_units()
```

## Installation

```bash
python -m pip install torch-weighttracker
```

BOPs MAC accounting uses `fvcore` for baseline per-module MACs:

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
- Track active structured or unstructured BOPs and compression rate during
  pruning, sparsity-aware training, or quantization-aware training (QAT).
- Inspect which modules participate in each channel, feature, head, or head-dim
  group.
- Build structural metrics that aggregate many weight tensors into one value per
  pruning unit.
- Physically prune zeroed canonical units, including attention heads, after
  sparsity-aware training.

## Current Pruning Notes

`WeightTracker` can inspect zeroed canonical units with `view_zero_units()` /
`view_zero_structures()` and physically remove them with `prune_zero_units()` /
`prune_zero_structures()`. You can also remove one canonical unit directly with
`prune_unit(group_id, unit_id)`.

Zero detection can ignore module instances or module types, matching tracker
filter semantics. The ignore filter only decides whether a structure is zero; if
that structure is pruned, the coupled Torch-Pruning group is still applied:

```python
zero_view = tracker.view_zero_structures(ignore=[torch.nn.BatchNorm2d])
tracker.prune_zero_structures(ignore=[torch.nn.BatchNorm2d])
```

Physical pruning changes module shapes and rebuilds the dependency state. Any
registered trackers or regularizers are cleared after `prune_unit()` or
`prune_zero_units()` / `prune_zero_structures()`, so recreate them before
collecting metrics or losses from the pruned model:

```python
metrics_before = tracker.create_tracker(
    "structured_bops",
    log_total_bops=True,
).track()

tracker.prune_zero_units()

metrics_after = tracker.create_tracker(
    "structured_bops",
    log_total_bops=True,
).track()
```

Fake pruning remains useful during training because it zeros the selected
canonical unit while keeping the module shapes intact:

```python
tracker.create_tracker("structured_bops", log_total_bops=True)
metrics_before = tracker.track()

tracker.fake_prune_unit(group_id=3, unit_id=0)
tracker.fake_prune_unit(group_id=3, unit_id=2)

metrics_after = tracker.track()
```

## timm ViT Attention Heads

The `torch_weighttracker.integrations.timm` helpers make timm ViT attention
blocks visible as head-level pruning groups. `infer_vit_num_heads(model)` maps
each fused `Attention.qkv` projection to its current head count, and
`sync_vit_attention_metadata` updates timm attention metadata after physical
head pruning.

```python
import timm
import torch

from torch_weighttracker import WeightTracker
from torch_weighttracker.integrations.timm import (
    infer_vit_num_heads,
    sync_vit_attention_metadata,
)

example_inputs = torch.rand(1, 3, 224, 224)
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=10,
)

tracker = WeightTracker(
    model,
    example_inputs=example_inputs,
    num_heads=infer_vit_num_heads(model),
    prune_num_heads=True,
    post_prune_hooks=(sync_vit_attention_metadata,),
)

print(tracker.view_structures())

tracker.create_tracker("structured_bops", log_total_bops=True)
metrics_before = tracker.track()

# Example: zero two attention heads in the group reported by view_structures().
tracker.fake_prune_unit(group_id=3, unit_id=0)
tracker.fake_prune_unit(group_id=3, unit_id=2)
metrics_after_fake_prune = tracker.track()

# Convert zeroed units into real shape changes, then recreate metric trackers.
tracker.prune_zero_units()
metrics_after_physical_prune = tracker.create_tracker(
    "structured_bops",
    log_total_bops=True,
).track()

print(metrics_before["structured_bops"])
print(metrics_after_fake_prune["structured_bops"])
print(metrics_after_physical_prune["structured_bops"])
```

For timm ViTs, head pruning removes complete q/k/v head slices from the fused
`qkv` projection and the corresponding projection input channels. The sync hook
keeps `num_heads`, `attn_dim`, `head_dim`, and `scale` consistent with the new
shape so the pruned model can still run a forward pass.


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
tracker.create_tracker(
    [TrackerType.STRUCTURED_BOPS, "group_pruning_summary"]
)
metrics = tracker.track()
```

#### Formulation of the Structured BOPs Metric

For each weighted module $m$, WeightTracker multiplies the active structured MAC
count by that module's activation and weight bit widths [1]:

$$
\mathit{StructuredBOPs}_m =
\mathit{ActiveMACs}_m
\cdot
b^{\mathrm{act}}_m
\cdot
b^{\mathrm{weight}}_m
$$

The active MAC count scales the dense module MAC count by the active fraction of
each structural cost axis:

$$
\mathit{ActiveMACs}_m =
\mathit{BaselineMACs}_m
\cdot
\prod_{a \in A_m}
\frac{n^{\mathrm{active}}_{m,a}}{n^{\mathrm{baseline}}_{m,a}}
$$

Compression is reported against a dense 32-bit activation and 32-bit weight
baseline:

$$
\mathit{BaselineBOPs}_m =
\mathit{BaselineMACs}_m
\cdot
32
\cdot
32
$$

$$
\mathit{CompressionRate} =
1 -
\frac{\sum_m \mathit{StructuredBOPs}_m}
{\sum_m \mathit{BaselineBOPs}_m}
$$

Where:

- $\mathit{StructuredBOPs}_m$: active bit operations for weighted module
  $m$.
- $\mathit{ActiveMACs}_m$: active MAC count after structured units are
  masked or pruned.
- $\mathit{BaselineMACs}_m$: dense MAC count for module $m$ before
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

## Unstructured BOPs

Unstructured BOPs combines each layer's dense runtime MAC count with its
active weight fraction and activation/weight bit widths:

```python
import torch

from torch_weighttracker.trackers import TrackerType

metrics = tracker.create_tracker(
    TrackerType.UNSTRUCTURED_BOPS,
    include=[model.layer3, model.layer4],
    ignore=[torch.nn.BatchNorm2d],
).track()

print(metrics["unstructured_bops_compression"])

raw_metrics = tracker.create_tracker(
    TrackerType.UNSTRUCTURED_BOPS,
    include=[model.layer3, model.layer4],
    ignore=[torch.nn.BatchNorm2d],
    log_total_bops=True,
    log_layerwise_stats=True,
).track()

print(raw_metrics["unstructured_bops"])
print(raw_metrics["unstructured_bops_pr_module"])
print(raw_metrics["unstructured_bops_compression_rate_pr_module"])
```

For each weighted module $m$:

$$
\mathit{UnstructuredBOPs}_m =
\mathit{BaselineMACs}_m
\cdot
(1 - \mathit{Sparsity}_m)
\cdot
b^{\mathrm{act}}_m
\cdot
b^{\mathrm{weight}}_m
$$

Compression uses the same dense 32-bit baseline as structured BOPs:

$$
\mathit{CompressionRate} =
1 -
\frac{\sum_m \mathit{UnstructuredBOPs}_m}
{\sum_m \mathit{BaselineMACs}_m \cdot 32 \cdot 32}
$$

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

## Group Pruning Summary

Group pruning summary reports pruned canonical units and group-attributed
pruned parameters as flat scalar keys that can be passed directly to loggers
such as W&B:

```python
import torch

from torch_weighttracker.trackers import TrackerType

metrics = tracker.create_tracker(
    TrackerType.GROUP_PRUNING_SUMMARY,
    include=[model.layer3, model.layer4],
    ignore=[torch.nn.BatchNorm2d],
).track()

print(metrics["group_pruning/pruned_units"])
print(metrics["group_pruning/pruned_params"])
```

Per-group values are emitted under keys such as
`group_pruning/groups/layer3.0.conv1:prune_out_channels/pruned_units` and
`group_pruning/groups/layer3.0.conv1:prune_out_channels/pruned_params`.

## Architecture

The main API is `WeightTracker`. Internally it is split into a few layers:

1. Dependency discovery: `WeightTracker` builds dependency groups from the model
   and `example_inputs` using Torch-Pruning's dependency graph machinery [2],
   whose work we gratefully build on.
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

[2] Fang et al., [Torch-Pruning](https://github.com/VainF/Torch-Pruning).
