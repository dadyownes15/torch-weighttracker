from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from torch_weighttracker.canonical_units import CanonicalUnitGroup, UnitKind


def group_names(owner, groups: Iterable[CanonicalUnitGroup]) -> tuple[str, ...]:
    groups = tuple(groups)
    base_names = tuple(_base_group_name(owner, group) for group in groups)
    counts = Counter(base_names)

    return tuple(
        (
            base_name
            if counts[base_name] == 1
            else f"{base_name}#{group.group_id}"
        )
        for base_name, group in zip(base_names, groups, strict=True)
    )


def _base_group_name(owner, group: CanonicalUnitGroup) -> str:
    if len(group.members) == 0:
        return f"group_{group.group_id}"

    root = group.members[0]
    module_name = owner._module_names_for_modules((root.module,))[0]
    name = f"{module_name}:{_handler_name(root.handler)}"

    if group.unit_kind != UnitKind.CHANNEL:
        name = f"{name}:{group.unit_kind.value}"

    return name


def _handler_name(handler) -> str:
    name = getattr(handler, "__name__", str(handler))
    if name.endswith("_out_channels"):
        return "prune_out_channels"
    if name.endswith("_in_channels"):
        return "prune_in_channels"
    return name
