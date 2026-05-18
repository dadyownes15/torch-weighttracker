from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import torch.nn as nn

from torch_weighttracker.canonical_units import CanonicalUnitGroup

FilterItem = nn.Module | type[nn.Module]
IgnoreItem = FilterItem


class _ModuleMatcher:
    def __init__(self, items: Iterable[FilterItem]) -> None:
        modules: list[nn.Module] = []
        module_types: list[type[nn.Module]] = []

        for item in items:
            if isinstance(item, nn.Module):
                modules.extend(item.modules())
            elif isinstance(item, type) and issubclass(item, nn.Module):
                module_types.append(item)
            else:
                raise TypeError(
                    "include and ignore entries must be nn.Module instances or "
                    "nn.Module types."
                )

        self.modules = frozenset(modules)
        self.module_types = tuple(module_types)

    def __bool__(self) -> bool:
        return bool(self.modules or self.module_types)

    def matches(self, module: nn.Module) -> bool:
        return module in self.modules or isinstance(module, self.module_types)


class ConsumerFilter:
    def __init__(
        self,
        *,
        include: Iterable[FilterItem] = (),
        ignore: Iterable[FilterItem] = (),
    ) -> None:
        self.include = _ModuleMatcher(include)
        self.ignore = _ModuleMatcher(ignore)

    def __bool__(self) -> bool:
        return bool(self.include or self.ignore)

    def allows(self, module: nn.Module) -> bool:
        if self.include and not self.include.matches(module):
            return False

        return not self.ignore.matches(module)


def filter_canonical_members(
    groups: Iterable[CanonicalUnitGroup],
    filters: ConsumerFilter,
) -> tuple[CanonicalUnitGroup, ...]:
    return tuple(
        replace(
            group,
            members=tuple(
                member
                for member in group.members
                if filters.allows(member.module)
            ),
        )
        for group in groups
    )


def filter_modules(
    modules: Iterable[nn.Module],
    filters: ConsumerFilter,
) -> tuple[nn.Module, ...]:
    return tuple(module for module in modules if filters.allows(module))
