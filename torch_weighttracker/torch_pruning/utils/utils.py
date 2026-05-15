"""Minimal utility helpers used by the vendored dependency graph."""

import torch


def flatten_as_list(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        flattened = []
        for item in obj:
            flattened.extend(flatten_as_list(item))
        return flattened
    if isinstance(obj, dict):
        flattened = []
        for item in obj.values():
            flattened.extend(flatten_as_list(item))
        return flattened
    return obj
