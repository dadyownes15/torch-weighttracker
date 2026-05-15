from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import torch.nn as nn

from torch_weighttracker.canonical_units import CanonicalUnitGroup


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


def without_ignored_canonical_members(
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
