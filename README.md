<<<<<<< HEAD
# torch-structure-analyser

Local Python package for analyzing structured sparsity in PyTorch models.

## Local install

```bash
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e .[dev]
```

## Smoke test

```bash
python -c "import torch_structure_analyser as tsa; print(tsa.__version__)"
```

# torch-structure-analyzer
=======
# torch-structracker

`torch-structracker` is a PyPI-ready Python package scaffold for PyTorch-based
structured state tracking.

## Installation

```bash
pip install torch-structracker
```

For local development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Usage

```python
from torch_structracker import StructTracker

tracker = StructTracker()
tracker.update("layer_1", {"active": True})

print(tracker.get("layer_1"))
```

## Development

Run tests:

```bash
pytest
```

Run linting and formatting checks:

```bash
ruff check .
ruff format --check .
```

Build distributions:

```bash
python -m build
```

Publish with Twine after validating the generated artifacts:

```bash
twine check dist/*
twine upload dist/*
```

>>>>>>> gpu-based-group-lasso-generic
