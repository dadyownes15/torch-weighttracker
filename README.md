# torch-structracker

Tools for analyzing, tracking, and pruning structured sparsity in PyTorch
models.

## Local Install

```bash
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

Run linting and formatting checks:

```bash
ruff check .
ruff format --check .
```

## Smoke Tests

```bash
python -c "from torch_structracker.structure_tracker import StructureTracker; print(StructureTracker)"
```
