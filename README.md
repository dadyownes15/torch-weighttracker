# torch-weighttracker

PyTorch tools for tracking structured weight sparsity, regularization signals,
and bit-operation estimates in neural network modules.

The public API is centered on `WeightTracker`:

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

## Development

```bash
python -m pip install -e ".[dev]"
```

Run tests and lint checks:

```bash
pytest
ruff check .
ruff format --check .
```

## Smoke Test

```bash
python -c "from torch_weighttracker import WeightTracker; print(WeightTracker)"
```

## Status

This package is pre-1.0. Public APIs may still change while the tracker,
calculation, and regularizer surfaces settle.

## License

MIT
