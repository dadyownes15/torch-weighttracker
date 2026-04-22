from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from ..pruner import function
from .types import StructureAxis


class StructureRule:
    axis: StructureAxis

    def extract_tensors(
        self,
        module,
        local_idxs: tuple[int, ...],
        include_bias: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def extract_masks(
        self,
        module,
        local_idxs: tuple[int, ...],
        include_bias: bool = False,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        raise NotImplementedError

    def _sorted_unique(self, local_idxs: tuple[int, ...]) -> list[int]:
        return sorted(set(int(idx) for idx in local_idxs))


class ConvOutRule(StructureRule):
    axis = StructureAxis.OUT

    def extract_tensors(self, module, local_idxs, include_bias=False):
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        tensors = []
        if getattr(module, "transposed", False):
            tensors.append(module.weight[:, idxs, ...].reshape(-1))
        else:
            tensors.append(module.weight[idxs, ...].reshape(-1))
        if include_bias and module.bias is not None:
            tensors.append(module.bias[idxs].reshape(-1))
        return tuple(tensors)

    def extract_masks(self, module, local_idxs, include_bias=False):
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        masks = []
        weight_mask = torch.zeros_like(module.weight, dtype=torch.bool)
        if getattr(module, "transposed", False):
            weight_mask[:, idxs, ...] = True
        else:
            weight_mask[idxs, ...] = True
        masks.append((module.weight, weight_mask))
        if include_bias and module.bias is not None:
            bias_mask = torch.zeros_like(module.bias, dtype=torch.bool)
            bias_mask[idxs] = True
            masks.append((module.bias, bias_mask))
        return tuple(masks)


class ConvInRule(StructureRule):
    axis = StructureAxis.IN

    def _normalize_local_idxs(self, module, local_idxs: tuple[int, ...]) -> list[int]:
        idxs = self._sorted_unique(local_idxs)
        if module.groups > 1 and module.groups != module.in_channels:
            relative_width = module.in_channels // module.groups
            idxs = sorted({idx % relative_width for idx in idxs})
        return idxs

    def extract_tensors(self, module, local_idxs, include_bias=False):
        idxs = self._normalize_local_idxs(module, local_idxs)
        if len(idxs) == 0:
            return ()
        if getattr(module, "transposed", False):
            return (module.weight[idxs, ...].reshape(-1),)
        return (module.weight[:, idxs, ...].reshape(-1),)

    def extract_masks(self, module, local_idxs, include_bias=False):
        idxs = self._normalize_local_idxs(module, local_idxs)
        if len(idxs) == 0:
            return ()
        weight_mask = torch.zeros_like(module.weight, dtype=torch.bool)
        if getattr(module, "transposed", False):
            weight_mask[idxs, ...] = True
        else:
            weight_mask[:, idxs, ...] = True
        return ((module.weight, weight_mask),)


class LinearOutRule(StructureRule):
    axis = StructureAxis.OUT

    def extract_tensors(self, module, local_idxs, include_bias=False):
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        tensors = [module.weight[idxs, :].reshape(-1)]
        if include_bias and module.bias is not None:
            tensors.append(module.bias[idxs].reshape(-1))
        return tuple(tensors)

    def extract_masks(self, module, local_idxs, include_bias=False):
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        masks = []
        weight_mask = torch.zeros_like(module.weight, dtype=torch.bool)
        weight_mask[idxs, :] = True
        masks.append((module.weight, weight_mask))
        if include_bias and module.bias is not None:
            bias_mask = torch.zeros_like(module.bias, dtype=torch.bool)
            bias_mask[idxs] = True
            masks.append((module.bias, bias_mask))
        return tuple(masks)


class LinearInRule(StructureRule):
    axis = StructureAxis.IN

    def extract_tensors(self, module, local_idxs, include_bias=False):
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        return (module.weight[:, idxs].reshape(-1),)

    def extract_masks(self, module, local_idxs, include_bias=False):
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        weight_mask = torch.zeros_like(module.weight, dtype=torch.bool)
        weight_mask[:, idxs] = True
        return ((module.weight, weight_mask),)


class BatchNormOutRule(StructureRule):
    axis = StructureAxis.OUT

    def extract_tensors(self, module, local_idxs, include_bias=False):
        if not getattr(module, "affine", False):
            return ()
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        tensors = [module.weight[idxs].reshape(-1)]
        if include_bias and module.bias is not None:
            tensors.append(module.bias[idxs].reshape(-1))
        return tuple(tensors)

    def extract_masks(self, module, local_idxs, include_bias=False):
        if not getattr(module, "affine", False):
            return ()
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        masks = []
        weight_mask = torch.zeros_like(module.weight, dtype=torch.bool)
        weight_mask[idxs] = True
        masks.append((module.weight, weight_mask))
        if include_bias and module.bias is not None:
            bias_mask = torch.zeros_like(module.bias, dtype=torch.bool)
            bias_mask[idxs] = True
            masks.append((module.bias, bias_mask))
        return tuple(masks)


class LayerNormOutRule(StructureRule):
    axis = StructureAxis.OUT

    def extract_tensors(self, module, local_idxs, include_bias=False):
        if not getattr(module, "elementwise_affine", False):
            return ()
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        tensors = [module.weight[idxs].reshape(-1)]
        if include_bias and module.bias is not None:
            tensors.append(module.bias[idxs].reshape(-1))
        return tuple(tensors)

    def extract_masks(self, module, local_idxs, include_bias=False):
        if not getattr(module, "elementwise_affine", False):
            return ()
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        masks = []
        weight_mask = torch.zeros_like(module.weight, dtype=torch.bool)
        weight_mask[idxs] = True
        masks.append((module.weight, weight_mask))
        if include_bias and module.bias is not None:
            bias_mask = torch.zeros_like(module.bias, dtype=torch.bool)
            bias_mask[idxs] = True
            masks.append((module.bias, bias_mask))
        return tuple(masks)


class GroupNormOutRule(StructureRule):
    axis = StructureAxis.OUT

    def extract_tensors(self, module, local_idxs, include_bias=False):
        if not getattr(module, "affine", False):
            return ()
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        tensors = [module.weight[idxs].reshape(-1)]
        if include_bias and module.bias is not None:
            tensors.append(module.bias[idxs].reshape(-1))
        return tuple(tensors)

    def extract_masks(self, module, local_idxs, include_bias=False):
        if not getattr(module, "affine", False):
            return ()
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        masks = []
        weight_mask = torch.zeros_like(module.weight, dtype=torch.bool)
        weight_mask[idxs] = True
        masks.append((module.weight, weight_mask))
        if include_bias and module.bias is not None:
            bias_mask = torch.zeros_like(module.bias, dtype=torch.bool)
            bias_mask[idxs] = True
            masks.append((module.bias, bias_mask))
        return tuple(masks)


class InstanceNormOutRule(GroupNormOutRule):
    pass


class EmbeddingOutRule(StructureRule):
    axis = StructureAxis.OUT

    def extract_tensors(self, module, local_idxs, include_bias=False):
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        return (module.weight[:, idxs].reshape(-1),)

    def extract_masks(self, module, local_idxs, include_bias=False):
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        weight_mask = torch.zeros_like(module.weight, dtype=torch.bool)
        weight_mask[:, idxs] = True
        return ((module.weight, weight_mask),)


class UnwrappedParameterRule(StructureRule):
    axis = StructureAxis.PARAM

    def __init__(self, pruning_dim_getter: Callable[[torch.nn.Parameter], int]):
        self._pruning_dim_getter = pruning_dim_getter

    def extract_tensors(self, module, local_idxs, include_bias=False):
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        pruning_dim = self._pruning_dim_getter(module)
        indexer = [slice(None)] * module.ndim
        indexer[pruning_dim] = idxs
        return (module[tuple(indexer)].reshape(-1),)

    def extract_masks(self, module, local_idxs, include_bias=False):
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        pruning_dim = self._pruning_dim_getter(module)
        mask = torch.zeros_like(module, dtype=torch.bool)
        indexer = [slice(None)] * module.ndim
        indexer[pruning_dim] = idxs
        mask[tuple(indexer)] = True
        return ((module, mask),)


class PReLUOutRule(StructureRule):
    axis = StructureAxis.OUT

    def extract_tensors(self, module, local_idxs, include_bias=False):
        if module.num_parameters == 1:
            return ()
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        return (module.weight[idxs].reshape(-1),)

    def extract_masks(self, module, local_idxs, include_bias=False):
        if module.num_parameters == 1:
            return ()
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()
        weight_mask = torch.zeros_like(module.weight, dtype=torch.bool)
        weight_mask[idxs] = True
        return ((module.weight, weight_mask),)


class MultiheadAttentionOutRule(StructureRule):
    axis = StructureAxis.OUT

    def extract_tensors(self, module, local_idxs, include_bias=False):
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()

        tensors = []
        embed_dim = module.embed_dim

        def _select_masked_values(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return tensor[mask]

        if module.q_proj_weight is not None:
            row_mask = torch.zeros_like(module.q_proj_weight, dtype=torch.bool)
            row_mask[idxs, :] = True
            tensors.append(_select_masked_values(module.q_proj_weight, row_mask))
        if module.k_proj_weight is not None:
            row_mask = torch.zeros_like(module.k_proj_weight, dtype=torch.bool)
            row_mask[idxs, :] = True
            tensors.append(_select_masked_values(module.k_proj_weight, row_mask))
        if module.v_proj_weight is not None:
            row_mask = torch.zeros_like(module.v_proj_weight, dtype=torch.bool)
            row_mask[idxs, :] = True
            tensors.append(_select_masked_values(module.v_proj_weight, row_mask))

        repeated_idxs = idxs + [idx + embed_dim for idx in idxs] + [idx + 2 * embed_dim for idx in idxs]
        if module.in_proj_weight is not None:
            mask = torch.zeros_like(module.in_proj_weight, dtype=torch.bool)
            mask[repeated_idxs, :] = True
            mask[:, idxs] = True
            tensors.append(_select_masked_values(module.in_proj_weight, mask))
        if include_bias and module.in_proj_bias is not None:
            tensors.append(module.in_proj_bias[repeated_idxs].reshape(-1))

        if module.bias_k is not None:
            mask = torch.zeros_like(module.bias_k, dtype=torch.bool)
            mask[:, :, idxs] = True
            tensors.append(_select_masked_values(module.bias_k, mask))
        if module.bias_v is not None:
            mask = torch.zeros_like(module.bias_v, dtype=torch.bool)
            mask[:, :, idxs] = True
            tensors.append(_select_masked_values(module.bias_v, mask))

        if module.out_proj is not None:
            mask = torch.zeros_like(module.out_proj.weight, dtype=torch.bool)
            mask[idxs, :] = True
            mask[:, idxs] = True
            tensors.append(_select_masked_values(module.out_proj.weight, mask))
            if include_bias and module.out_proj.bias is not None:
                tensors.append(module.out_proj.bias[idxs].reshape(-1))

        return tuple(tensors)

    def extract_masks(self, module, local_idxs, include_bias=False):
        idxs = self._sorted_unique(local_idxs)
        if len(idxs) == 0:
            return ()

        masks = []
        embed_dim = module.embed_dim
        repeated_idxs = idxs + [idx + embed_dim for idx in idxs] + [idx + 2 * embed_dim for idx in idxs]

        if module.q_proj_weight is not None:
            row_mask = torch.zeros_like(module.q_proj_weight, dtype=torch.bool)
            row_mask[idxs, :] = True
            masks.append((module.q_proj_weight, row_mask))
        if module.k_proj_weight is not None:
            row_mask = torch.zeros_like(module.k_proj_weight, dtype=torch.bool)
            row_mask[idxs, :] = True
            masks.append((module.k_proj_weight, row_mask))
        if module.v_proj_weight is not None:
            row_mask = torch.zeros_like(module.v_proj_weight, dtype=torch.bool)
            row_mask[idxs, :] = True
            masks.append((module.v_proj_weight, row_mask))

        if module.in_proj_weight is not None:
            mask = torch.zeros_like(module.in_proj_weight, dtype=torch.bool)
            mask[repeated_idxs, :] = True
            mask[:, idxs] = True
            masks.append((module.in_proj_weight, mask))
        if include_bias and module.in_proj_bias is not None:
            bias_mask = torch.zeros_like(module.in_proj_bias, dtype=torch.bool)
            bias_mask[repeated_idxs] = True
            masks.append((module.in_proj_bias, bias_mask))

        if module.bias_k is not None:
            mask = torch.zeros_like(module.bias_k, dtype=torch.bool)
            mask[:, :, idxs] = True
            masks.append((module.bias_k, mask))
        if module.bias_v is not None:
            mask = torch.zeros_like(module.bias_v, dtype=torch.bool)
            mask[:, :, idxs] = True
            masks.append((module.bias_v, mask))

        if module.out_proj is not None:
            mask = torch.zeros_like(module.out_proj.weight, dtype=torch.bool)
            mask[idxs, :] = True
            mask[:, idxs] = True
            masks.append((module.out_proj.weight, mask))
            if include_bias and module.out_proj.bias is not None:
                bias_mask = torch.zeros_like(module.out_proj.bias, dtype=torch.bool)
                bias_mask[idxs] = True
                masks.append((module.out_proj.bias, bias_mask))

        return tuple(masks)


@dataclass
class RuleRegistry:
    _rules: dict[Callable[..., object], StructureRule]

    def register(self, handler: Callable[..., object], rule: StructureRule) -> None:
        self._rules[handler] = rule

    def resolve(self, handler: Callable[..., object]) -> Optional[StructureRule]:
        return self._rules.get(handler)


def infer_axis(handler: Callable[..., object]) -> StructureAxis:
    name = handler.__name__
    if "head" in name and "dim" in name:
        return StructureAxis.HEAD_DIM
    if "head" in name:
        return StructureAxis.HEAD
    if "_in_" in name:
        return StructureAxis.IN
    if "parameter" in name:
        return StructureAxis.PARAM
    return StructureAxis.OUT


def build_default_rule_registry(
    parameter_pruning_dim_getter: Callable[[torch.nn.Parameter], int],
) -> RuleRegistry:
    registry = RuleRegistry(_rules={})
    registry.register(function.prune_conv_out_channels, ConvOutRule())
    registry.register(function.prune_depthwise_conv_out_channels, ConvOutRule())
    registry.register(function.prune_conv_in_channels, ConvInRule())
    registry.register(function.prune_depthwise_conv_in_channels, ConvInRule())
    registry.register(function.prune_linear_out_channels, LinearOutRule())
    registry.register(function.prune_linear_in_channels, LinearInRule())
    registry.register(function.prune_batchnorm_out_channels, BatchNormOutRule())
    registry.register(function.prune_batchnorm_in_channels, BatchNormOutRule())
    registry.register(function.prune_layernorm_out_channels, LayerNormOutRule())
    registry.register(function.prune_layernorm_in_channels, LayerNormOutRule())
    registry.register(function.prune_groupnorm_out_channels, GroupNormOutRule())
    registry.register(function.prune_groupnorm_in_channels, GroupNormOutRule())
    registry.register(function.prune_instancenorm_out_channels, InstanceNormOutRule())
    registry.register(function.prune_instancenorm_in_channels, InstanceNormOutRule())
    registry.register(function.prune_embedding_out_channels, EmbeddingOutRule())
    registry.register(function.prune_embedding_in_channels, EmbeddingOutRule())
    registry.register(function.prune_prelu_out_channels, PReLUOutRule())
    registry.register(function.prune_prelu_in_channels, PReLUOutRule())
    registry.register(function.prune_multihead_attention_out_channels, MultiheadAttentionOutRule())
    registry.register(function.prune_multihead_attention_in_channels, MultiheadAttentionOutRule())
    registry.register(function.prune_parameter_out_channels, UnwrappedParameterRule(parameter_pruning_dim_getter))
    registry.register(function.prune_parameter_in_channels, UnwrappedParameterRule(parameter_pruning_dim_getter))
    return registry
