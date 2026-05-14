# Consumer Ignore Spec

## API

```python
tracker.create_tracker(
    TrackerType.STRUCTURED_BOPS,
    ignore=[nn.GroupNorm],
)

tracker.create_regularizer(
    RegularizerType.GROUP_LASSO,
    ignore=[model.head, model.output_projection],
)
```

```python
class StructureTracker:
    def create_tracker(
        self,
        tracker_type: TrackerType,
        *,
        ignore: Iterable[nn.Module | type[nn.Module]] = (),
        **kwargs,
    ):
        tracker_type = TrackerType(tracker_type)
        tracker_cls = tracker_class_for_type(tracker_type)

        context = tracker_cls.calculation_context(
            self,
            ignore=ignore,
            **kwargs,
        )
        required_calculations = tracker_cls.required_calculations
        calculations = self.ensure_calculations(
            required_calculations,
            context=context,
        )
        tracker = tracker_cls(calculations=calculations, **kwargs)
        self.trackers.append(tracker)
        return tracker

    def create_regularizer(
        self,
        regularizer_type: RegularizerType,
        *,
        ignore: Iterable[nn.Module | type[nn.Module]] = (),
        **kwargs,
    ):
        regularizer_type = RegularizerType(regularizer_type)
        regularizer_cls = regularizer_class_for_type(regularizer_type)

        context = regularizer_cls.calculation_context(
            self,
            ignore=ignore,
            **kwargs,
        )
        required_calculations = regularizer_cls.required_calculations
        calculations = self.ensure_calculations(
            required_calculations,
            context=context,
        )
        regularizer = regularizer_cls(calculations=calculations, **kwargs)
        self.regularizers.append(regularizer)
        return regularizer
```

`ignore` is a calculation-context option and is not forwarded to the consumer
constructor. Other `**kwargs` are consumer constructor kwargs. Unsupported kwargs
should fail with the consumer's normal `TypeError`; they must not be silently
dropped by `StructureTracker`.

## Ignore Normalization

```python
from __future__ import annotations

from dataclasses import replace
from collections.abc import Iterable

import torch.nn as nn

from torch_structracker.calculations import CalcType, CachedCalculation
from torch_structracker.calculations.context import CalculationContext
import torch_structracker.calculations.calculations as calculation_impl
from torch_structracker.canonical_units import CanonicalMember, CanonicalUnitGroup
from torch_structracker.reductions.builder import IndexSelection, SegmentSelection


IgnoreItem = nn.Module | type[nn.Module]


class ModuleIgnore:
    def __init__(self, ignore: Iterable[IgnoreItem]) -> None:
        modules: list[nn.Module] = []
        module_types: list[type[nn.Module]] = []

        for item in ignore:
            if isinstance(item, nn.Module):
                modules.extend(item.modules())
            elif isinstance(item, type) and issubclass(item, nn.Module):
                module_types.append(item)
            else:
                raise TypeError(
                    "ignore entries must be nn.Module instances or nn.Module types."
                )

        self.modules = frozenset(modules)
        self.module_types = tuple(module_types)

    def __bool__(self) -> bool:
        return bool(self.modules or self.module_types)

    def matches(self, module: nn.Module) -> bool:
        return module in self.modules or isinstance(module, self.module_types)
```

## Consumer Hooks

```python
class BaseTracker(nn.Module, ABC):
    required_calculations: tuple[CalcType, ...] = ()

    @classmethod
    def calculation_context(
        cls,
        owner: StructureTracker,
        *,
        ignore: Iterable[IgnoreItem] = (),
        **kwargs,
    ) -> CalculationContext | None:
        return None
```

```python
class BaseRegularizer(nn.Module, ABC):
    required_calculations: tuple[CalcType, ...] = ()

    @classmethod
    def calculation_context(
        cls,
        owner: StructureTracker,
        *,
        ignore: Iterable[IgnoreItem] = (),
        **kwargs,
    ) -> CalculationContext | None:
        return None
```

`None` means: use the default calculation context and the existing `CalcType`
cache keys.

V1 keeps the required-calculation tuple static. `calculation_context(...)` may
change the structural inputs used to build those calculations, but it must not
change which calculations the consumer requires. If a future consumer genuinely
needs dynamic required calculations, that change must update `BaseTracker` and
`BaseRegularizer` validation to validate against the resolved tuple rather than
the class attribute.

## Context Creation

```python
class StructureTracker:
    def _calculation_context(
        self,
        *,
        canonical_groups: Iterable[CanonicalUnitGroup] | None = None,
        weighted_modules: Iterable[nn.Module] | None = None,
    ) -> CalculationContext:
        canonical_groups = (
            tuple(self.canonical_groups)
            if canonical_groups is None
            else tuple(canonical_groups)
        )
        weighted_modules = (
            self._get_weighted_modules()
            if weighted_modules is None
            else tuple(weighted_modules)
        )

        return CalculationContext(
            model=self.model,
            canonical_groups=canonical_groups,
            device=self.device,
            dtype=self.dtype,
            weighted_modules=weighted_modules,
            weighted_module_index={
                module: index for index, module in enumerate(weighted_modules)
            },
            example_inputs=self.example_inputs,
        )
```

## Calculation Cache

```python
class StructureTracker:
    def __init__(...):
        self.calculations: dict[CalcType | tuple[CalcType, tuple], nn.Module] = {}

    def ensure_calculations(
        self,
        calculation_types: tuple[CalcType, ...],
        *,
        context: CalculationContext | None = None,
    ) -> dict[CalcType, nn.Module]:
        context_key = None
        if context is None:
            context = self._calculation_context()
        else:
            context_key = self._calculation_context_key(context)

        calculations = {}
        for calculation_type in calculation_types:
            calculations[calculation_type] = self._get_calculation(
                calculation_type,
                context=context,
                context_key=context_key,
                stack=(),
            )
        return calculations

    def _get_calculation(
        self,
        calculation_type: CalcType,
        *,
        context: CalculationContext,
        context_key: tuple | None,
        stack: tuple[CalcType, ...],
    ) -> nn.Module:
        cache_key = self._calculation_cache_key(calculation_type, context_key)
        if cache_key not in self.calculations:
            self.calculations[cache_key] = self._create_calculation(
                calculation_type,
                context=context,
                context_key=context_key,
                stack=stack,
            )

        return self.calculations[cache_key]

    def _create_calculation(
        self,
        calculation_type: CalcType,
        *,
        context: CalculationContext,
        context_key: tuple | None,
        stack: tuple[CalcType, ...],
    ) -> nn.Module:
        if calculation_type in stack:
            cycle = (*stack, calculation_type)
            names = " -> ".join(item.value for item in cycle)
            raise ValueError(f"Circular calculation dependency detected: {names}")

        try:
            spec = calculation_impl.CALCULATION_SPECS[calculation_type]
        except KeyError as error:
            raise ValueError(
                f"Unknown calculation type: {calculation_type.value}"
            ) from error

        if spec.requires_groups and len(context.canonical_groups) == 0:
            raise ValueError(
                f"{calculation_type.value} requires dependency groups. Pass groups "
                "when constructing StructureTracker."
            )

        next_stack = (*stack, calculation_type)
        dependencies = {
            dependency: self._get_calculation(
                dependency,
                context=context,
                context_key=context_key,
                stack=next_stack,
            )
            for dependency in spec.required_calculations
        }
        calculation = spec.create(context, dependencies)

        if spec.cache_constant:
            cached = CachedCalculation(calculation)
            cached.refresh_cache()
            return cached

        return calculation

    def _calculation_cache_key(
        self,
        calculation_type: CalcType,
        context_key: tuple | None,
    ) -> CalcType | tuple[CalcType, tuple]:
        if context_key is None:
            return calculation_type
        return calculation_type, context_key

    def _calculation_context_key(
        self,
        context: CalculationContext,
    ) -> tuple:
        return (
            tuple(_canonical_group_key(group) for group in context.canonical_groups),
            tuple(id(module) for module in context.weighted_modules),
            context.device,
            context.dtype,
            None if context.example_inputs is None else id(context.example_inputs),
        )
```

```python
def _canonical_group_key(group: CanonicalUnitGroup) -> tuple:
    return (
        group.group_id,
        group.offset,
        group.length,
        group.unit_kind,
        tuple(_canonical_member_key(member) for member in group.members),
    )


def _canonical_member_key(member: CanonicalMember) -> tuple:
    return (
        id(member.module),
        member.unit_axis,
        member.source_layout,
        _selection_key(member.destination),
        member.source_indices,
        member.embed_dim,
        member.num_heads,
        member.head_dim,
    )


def _selection_key(selection: SegmentSelection | IndexSelection) -> tuple:
    if isinstance(selection, SegmentSelection):
        return "segment", int(selection.start), int(selection.length)
    if isinstance(selection, IndexSelection):
        return "index", tuple(int(index) for index in selection.indices)
    raise TypeError(f"Unsupported selection type: {type(selection).__name__}")
```

## Structured BOPs

```python
class StructuredBOPs(BaseTracker):
    required_calculations = (
        CalcType.ACTIVE_MACS_PR_MODULE,
        CalcType.BITRATE_PR_MODULE,
    )

    @classmethod
    def calculation_context(
        cls,
        owner: StructureTracker,
        *,
        ignore: Iterable[IgnoreItem] = (),
        **kwargs,
    ) -> CalculationContext | None:
        ignored = ModuleIgnore(ignore)
        if not ignored:
            return None

        weighted_modules = tuple(
            module
            for module in owner._get_weighted_modules()
            if not ignored.matches(module)
        )
        return owner._calculation_context(
            canonical_groups=_without_ignored_canonical_members(
                owner.canonical_groups,
                ignored,
            ),
            weighted_modules=weighted_modules,
        )
```

Effect:

```text
ACTIVE_UNITS               -> filtered canonical members
UNIT_ACTIVE_MASK           -> filtered canonical members
BITRATE_PR_MODULE           -> filtered weighted module list
BASELINE_MACS_PR_MODULE    -> filtered weighted module list
BASELINE_MODULE_AXES       -> filtered weighted module list
UNIT_DELTA_TO_MODULE_AXIS  -> filtered weighted_module_index
ACTIVE_MACS_PR_MODULE      -> filtered module output
```

## Group Lasso

```python
class GroupLasso(BaseRegularizer):
    required_calculations = (
        CalcType.L2_NORM_PR_UNIT,
        CalcType.UNITS_TO_GROUP,
        CalcType.UNIT_ACTIVE_MASK,
        CalcType.BASELINE_GROUP_SIZES,
        CalcType.GROUP_CHANGE_EFFECT,
        CalcType.GROUP_SIZES,
    )

    @classmethod
    def calculation_context(
        cls,
        owner: StructureTracker,
        *,
        ignore: Iterable[IgnoreItem] = (),
        **kwargs,
    ) -> CalculationContext | None:
        ignored = ModuleIgnore(ignore)
        if not ignored:
            return None

        return owner._calculation_context(
            canonical_groups=_without_ignored_canonical_members(
                owner.canonical_groups,
                ignored,
            )
        )
```

```python
def _without_ignored_canonical_members(
    groups: Iterable[CanonicalUnitGroup],
    ignored: ModuleIgnore,
) -> tuple[CanonicalUnitGroup, ...]:
    return tuple(
        replace(
            group,
            members=tuple(
                member
                for member in group.members
                if not ignored.matches(member.module)
            ),
        )
        for group in groups
    )
```

Effect:

```text
group offsets stay stable
group lengths stay stable
empty groups stay present
L2_NORM_PR_UNIT ignores filtered member parameters
ACTIVE_UNITS ignores filtered member parameters
GROUP_CHANGE_EFFECT ignores filtered member parameters
UNITS_TO_GROUP, BASELINE_GROUP_SIZES, GROUP_SIZES stay on the canonical unit space
```

## Empty Filtered Contexts

Per-consumer ignore can remove every canonical member from one or more groups.
That is not the same as constructing `StructureTracker` without dependency
groups.

Required behavior:

```text
groups may have members=()
group_id, offset, length, and unit_kind stay stable
len(context.canonical_groups) > 0 still satisfies requires_groups
zero filtered members produce zero-valued member-derived outputs
```

When all canonical members are filtered, member-derived calculations must return
typed zero tensors instead of asking `ReductionPlanBuilder` to finalize an empty
plan:

```text
ACTIVE_UNITS          -> zeros([total_units])
UNIT_ACTIVE_MASK      -> zeros([total_units])
L2_NORM_PR_UNIT       -> zeros([total_units])
STRUCTURED_UNIT_SUM   -> zeros([total_units])
GROUP_CHANGE_EFFECT   -> zeros([num_groups])
```

Group-index calculations still use the canonical unit space:

```text
UNITS_TO_GROUP         -> unchanged pipeline over group offsets and lengths
BASELINE_GROUP_SIZES   -> original canonical group lengths
GROUP_SIZES            -> original canonical group lengths
```

For `StructuredBOPs`, ignoring every weighted module is valid. Module-derived
calculations should return empty tensors with the normal rank/layout rather than
raising from empty module-plan construction:

```text
BITRATE_PR_MODULE          -> shape [0, 2] or flat [0]
BASELINE_MACS_PR_MODULE    -> shape [0]
BASELINE_MODULE_AXES       -> shape [0, 2] or flat [0]
UNIT_DELTA_TO_MODULE_AXIS  -> shape [0, 2] or flat [0]
ACTIVE_MACS_PR_MODULE      -> shape [0]
```

The tracker metric sum over an empty per-module tensor is zero.

## Construction-Time Structural Exclusion

```python
tracker = StructureTracker(
    model,
    example_inputs=example_inputs,
    ignored_layers=[model.unstructured_head],
)
```

Existing behavior stays:

```text
ignored_layers is applied while building torch-pruning groups
ignored_layers removes members before canonicalization
ignored_layers changes tracker.canonical_groups globally
ignored_layers is not a per-consumer option
```

## Tests

Tests must prove ignore semantics, not only cache identity or tensor shape. At
least one test should mutate ignored weights and show that the consumer output
does not change, then mutate an unignored weight and show that it still does
change.

```python
def test_group_lasso_ignore_is_invariant_to_ignored_module_weights():
    tracker = StructureTracker(model, example_inputs=example_inputs)
    regularizer = tracker.create_regularizer(
        RegularizerType.GROUP_LASSO,
        ignore=[model.head],
    )

    before = regularizer()

    with torch.no_grad():
        model.head.weight.add_(1000.0)

    torch.testing.assert_close(regularizer(), before)

    with torch.no_grad():
        model.backbone.weight.add_(1.0)

    assert not torch.allclose(regularizer(), before)
```

```python
def test_group_lasso_ignore_all_members_returns_zero_penalty():
    tracker = StructureTracker(model, example_inputs=example_inputs)
    regularizer = tracker.create_regularizer(
        RegularizerType.GROUP_LASSO,
        ignore=[model],
    )

    penalty = regularizer()

    torch.testing.assert_close(
        penalty,
        torch.zeros((), dtype=penalty.dtype, device=penalty.device),
    )
```

```python
def test_structured_bops_ignore_all_weighted_modules_returns_empty_metric():
    tracker = StructureTracker(model, example_inputs=example_inputs)
    structured_bops = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        ignore=[model],
    )

    per_module = structured_bops.compute()

    assert per_module.shape == (0,)
    torch.testing.assert_close(
        structured_bops.track()["structured_bops"],
        torch.zeros((), dtype=per_module.dtype, device=per_module.device),
    )
```

```python
def test_structured_bops_ignore_filters_weighted_modules():
    tracker = StructureTracker(model, example_inputs=example_inputs)

    full = tracker.create_tracker(TrackerType.STRUCTURED_BOPS)
    filtered = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        ignore=[nn.GroupNorm],
    )

    assert full.compute().numel() == len(tracker._get_weighted_modules())
    assert filtered.compute().numel() == sum(
        not isinstance(module, nn.GroupNorm)
        for module in tracker._get_weighted_modules()
    )
```

```python
def test_group_lasso_ignore_uses_context_key():
    tracker = StructureTracker(model, example_inputs=example_inputs)
    tracker.ensure_calculations((CalcType.L2_NORM_PR_UNIT,))
    global_l2 = tracker.calculations[CalcType.L2_NORM_PR_UNIT]

    regularizer = tracker.create_regularizer(
        RegularizerType.GROUP_LASSO,
        ignore=[model.head],
    )

    assert regularizer.calc(CalcType.L2_NORM_PR_UNIT) is not global_l2
```

```python
def test_filtering_keeps_group_index_space_stable():
    tracker = StructureTracker(model, example_inputs=example_inputs)

    filtered_groups = _without_ignored_canonical_members(
        tracker.canonical_groups,
        ModuleIgnore([model.head]),
    )

    assert [
        (group.group_id, group.offset, group.length)
        for group in filtered_groups
    ] == [
        (group.group_id, group.offset, group.length)
        for group in tracker.canonical_groups
    ]
    assert all(
        member.module is not model.head
        for group in filtered_groups
        for member in group.members
    )
```

```python
def test_consumer_ignore_does_not_mutate_global_calculations():
    tracker = StructureTracker(model, example_inputs=example_inputs)

    tracker.ensure_calculations((CalcType.BITRATE_PR_MODULE,))
    global_calc = tracker.calculations[CalcType.BITRATE_PR_MODULE]

    tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        ignore=[nn.GroupNorm],
    )

    assert tracker.calculations[CalcType.BITRATE_PR_MODULE] is global_calc
```

```python
def test_same_ignore_reuses_context_keyed_calculation():
    tracker = StructureTracker(model, example_inputs=example_inputs)

    first = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        ignore=[nn.GroupNorm],
    )
    second = tracker.create_tracker(
        TrackerType.STRUCTURED_BOPS,
        ignore=[nn.GroupNorm],
    )

    assert first.calc(CalcType.BITRATE_PR_MODULE) is second.calc(
        CalcType.BITRATE_PR_MODULE
    )
```

```python
def test_consumer_kwargs_are_not_silently_dropped():
    tracker = StructureTracker(model, example_inputs=example_inputs)

    with pytest.raises(TypeError):
        tracker.create_tracker(
            TrackerType.STRUCTURED_BOPS,
            unsupported_option=True,
        )
```
