# Torch Pruning Update

## Group root index count

`torch_pruning.dependency.group.Group` does not expose a dedicated attribute or property for the number of root indices.

Current behavior:

- `Group.__len__()` returns the number of dependency entries in the group, not the number of root indices.
- `root_idxs` is attached to each `GroupItem`, not to the `Group` itself.
- `root_idxs` is populated when the dependency graph builds the merged pruning group.

Effective way to get the root-index count:

```python
len(group[0].root_idxs)
```

In practice, this is usually equivalent for the pruning root:

```python
len(group[0].idxs)
```

That works because the root layer is initialized with `idxs == root_idxs`, while non-root items keep their own local `idxs` plus a mapped `root_idxs`.

Relevant code:

- `torch_structracker/torch_pruning/dependency/group.py`
- `torch_structracker/torch_pruning/_helpers.py`
- `torch_structracker/torch_pruning/dependency/graph.py`
- `torch_structracker/torch_pruning/pruner/algorithms/growing_reg_pruner.py`
