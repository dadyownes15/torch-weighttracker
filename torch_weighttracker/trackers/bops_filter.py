from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn

from torch_weighttracker.consumer_ignore import (
    ConsumerFilter,
    FilterItem,
    filter_modules,
)

_DEFAULT_IGNORED_TYPES: tuple[type[nn.Module], ...] = (
    nn.modules.batchnorm._BatchNorm,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.modules.instancenorm._InstanceNorm,
)

if hasattr(nn, "RMSNorm"):
    _DEFAULT_IGNORED_TYPES = (*_DEFAULT_IGNORED_TYPES, nn.RMSNorm)

DEFAULT_BOPS_IGNORED_MODULE_TYPES = _DEFAULT_IGNORED_TYPES


def bops_consumer_filter(
    *,
    include: Iterable[FilterItem] = (),
    ignore: Iterable[FilterItem] = (),
) -> ConsumerFilter:
    return ConsumerFilter(
        include=include,
        ignore=(*DEFAULT_BOPS_IGNORED_MODULE_TYPES, *tuple(ignore)),
    )


def filter_bops_weighted_modules(
    modules: Iterable[nn.Module],
    filters: ConsumerFilter,
) -> tuple[nn.Module, ...]:
    return filter_modules(modules, filters)
