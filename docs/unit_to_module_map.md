# Unit To Module Map

This note describes a simple calculation plan for mapping structured units back
to the modules they belong to.

The goal is different from member unit sum:

- member unit sum answers: "how much parameter value belongs to each unit?"
- unit-to-module map answers: "which module does each unit belong to?"

The output is a static mapping created from the pruning groups. It does not need
to run weight reductions.

## Target Behavior

Given pruning groups, assign every structured unit a global unit index. Then map
that global unit index to one or more module indices.

Conceptually:

```python
unit_to_module = torch.zeros(num_units, num_modules)

group_offset = 0

for group in groups:
    group_size = len(group[0].root_idxs)

    for member in group:
        module = member.dep.target.module
        module_index = module_index_for(module)

        for root_idx in member.root_idxs:
            unit_index = group_offset + int(root_idx)
            unit_to_module[unit_index, module_index] = 1

    group_offset += group_size
```

The resulting tensor has shape:

```python
(num_units, num_modules)
```

Each row represents a structured unit. Each column represents a module.

## Why This Is Useful

Once we have a unit-level value, the map can project it to module-level values.

For example, if `active_unit_param_count` has shape `(num_units,)`, then:

```python
module_param_count = active_unit_param_count @ unit_to_module
```

This produces shape `(num_modules,)`.

If each unit maps to exactly one module, every row has a single `1`. If a
structured unit participates in several modules, the row can contain multiple
`1`s.

## Plan Shape

This does not need a reducer plan because no module parameters are read at
runtime. A small static plan is enough:

```python
@dataclass(frozen=True)
class UnitToModuleMapPlan:
    num_units: int
    num_modules: int
    unit_indices: tuple[int, ...]
    module_indices: tuple[int, ...]
    module_names: tuple[str, ...] | None = None
```

The calculation can register these indices as buffers:

```python
class UnitToModuleMap(BaseCalculation):
    def __init__(self, plan: UnitToModuleMapPlan):
        super().__init__()
        self.num_units = plan.num_units
        self.num_modules = plan.num_modules

        self.register_buffer(
            "unit_indices",
            torch.tensor(plan.unit_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "module_indices",
            torch.tensor(plan.module_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "values",
            torch.ones(len(plan.unit_indices)),
            persistent=False,
        )
```

Then `forward()` can either return a dense matrix:

```python
def forward(self) -> torch.Tensor:
    out = self.values.new_zeros(self.num_units, self.num_modules)
    out[self.unit_indices, self.module_indices] = self.values
    return out
```

Or, if this becomes large, it can return a sparse COO tensor:

```python
def forward(self) -> torch.Tensor:
    indices = torch.stack((self.unit_indices, self.module_indices))
    return torch.sparse_coo_tensor(
        indices,
        self.values,
        size=(self.num_units, self.num_modules),
    )
```

The dense version is simpler and probably fine initially. The sparse version is
better if we expect many units and many modules.

## Compiler Shape

A direct compiler can follow the same group/member loop as member unit sum:

```python
def compile_unit_to_module_map_plan(groups) -> UnitToModuleMapPlan:
    module_to_index = {}
    unit_indices = []
    module_indices = []
    group_offset = 0

    for group in groups:
        group_size = len(group[0].root_idxs)

        for member in group:
            module = member.dep.target.module
            module_index = module_to_index.setdefault(module, len(module_to_index))

            for root_idx in member.root_idxs:
                unit_indices.append(group_offset + int(root_idx))
                module_indices.append(module_index)

        group_offset += group_size

    return UnitToModuleMapPlan(
        num_units=group_offset,
        num_modules=len(module_to_index),
        unit_indices=tuple(unit_indices),
        module_indices=tuple(module_indices),
    )
```

This compiler is intentionally static:

- no parameter extractor
- no operation type
- no reducer
- no runtime graph traversal

It just records the structural relationship between units and modules.

## Duplicate Entries

The compiler should decide how to handle duplicate `(unit_index, module_index)`
pairs.

Recommended first behavior:

```python
pairs: set[tuple[int, int]] = set()
```

Only emit a pair once. This makes the map binary: a unit either belongs to a
module or it does not.

If later we need weighted ownership, for example because a unit contributes
multiple independent parameter tensors to the same module, we should represent
that with explicit values rather than accidental duplicate pairs.

## Public API

Suggested usage:

```python
plan = compile_unit_to_module_map_plan(groups)
unit_to_module = UnitToModuleMap(plan)

matrix = unit_to_module()
module_values = unit_values @ matrix
```

This keeps the module map separate from weight reductions. That separation is
important: unit-to-module mapping is structural, while member unit sum is
parameter-derived.
