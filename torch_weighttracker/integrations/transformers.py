from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from torch_weighttracker.canonical_units import (
    CanonicalUnitGroup,
    SeparateQKVAttentionSpec,
)
from torch_weighttracker.torch_pruning.pruner.function import (
    prune_linear_in_channels,
    prune_linear_out_channels,
)
from torch_weighttracker.weight_tracker import WeightTracker


def bert_pruning_config(model: nn.Module) -> dict:
    """Return a WeightTracker config for BERT heads and FFN neurons.

    The adapter intentionally relies on the model-level ``prune_heads`` API so
    Hugging Face can maintain its original-head bookkeeping and serialized
    ``config.pruned_heads`` metadata. Transformers releases without that API
    must provide generic ``SeparateQKVAttentionSpec`` callbacks directly.
    """

    _require_bert_prune_heads(model)
    attention_specs, ffn_pairs = _bert_specs_and_ffn_pairs(model)
    return {
        "root_module_types": [nn.Linear],
        "attention_specs": attention_specs,
        "canonical_group_filter": _bert_target_group_filter(ffn_pairs),
        "post_prune_hooks": (_refresh_bert_pruning_config,),
    }


def _require_bert_prune_heads(model: nn.Module) -> None:
    if callable(getattr(model, "prune_heads", None)):
        return
    raise RuntimeError(
        "bert_pruning_config requires a Hugging Face BERT model exposing the "
        "model-level prune_heads API. Transformers v5 removed head pruning; "
        "use a compatible Transformers v4 release or construct generic "
        "SeparateQKVAttentionSpec callbacks explicitly."
    )


def _bert_specs_and_ffn_pairs(
    model: nn.Module,
) -> tuple[
    tuple[SeparateQKVAttentionSpec, ...],
    tuple[tuple[nn.Linear, nn.Linear], ...],
]:
    layers = _bert_encoder_layers(model)
    specs: list[SeparateQKVAttentionSpec] = []
    ffn_pairs: list[tuple[nn.Linear, nn.Linear]] = []

    for layer_index, layer in enumerate(layers):
        attention = getattr(layer, "attention", None)
        self_attention = getattr(attention, "self", None)
        attention_output = getattr(attention, "output", None)
        query = getattr(self_attention, "query", None)
        key = getattr(self_attention, "key", None)
        value = getattr(self_attention, "value", None)
        output_projection = getattr(attention_output, "dense", None)
        num_heads = getattr(self_attention, "num_attention_heads", None)

        if not isinstance(attention, nn.Module):
            raise TypeError(f"BERT layer {layer_index} has no attention module.")
        if not all(
            isinstance(module, nn.Linear)
            for module in (query, key, value, output_projection)
        ):
            raise TypeError(
                f"BERT layer {layer_index} must expose separate Linear query, "
                "key, value, and output projections."
            )
        if num_heads is None:
            raise TypeError(
                f"BERT layer {layer_index} has no num_attention_heads metadata."
            )

        specs.append(
            SeparateQKVAttentionSpec(
                query_projection=query,
                key_projection=key,
                value_projection=value,
                output_projection=output_projection,
                attention_module=attention,
                num_heads=int(num_heads),
                prune_heads=_bert_head_pruner(model, layer_index),
            )
        )

        intermediate = getattr(getattr(layer, "intermediate", None), "dense", None)
        output = getattr(getattr(layer, "output", None), "dense", None)
        if not isinstance(intermediate, nn.Linear) or not isinstance(output, nn.Linear):
            raise TypeError(
                f"BERT layer {layer_index} must expose Linear FFN projections."
            )
        ffn_pairs.append((intermediate, output))

    return tuple(specs), tuple(ffn_pairs)


def _bert_encoder_layers(model: nn.Module) -> tuple[nn.Module, ...]:
    base_model = getattr(model, "base_model", model)
    encoder = getattr(base_model, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise TypeError(
            "bert_pruning_config expects a BERT model with base_model.encoder.layer."
        )
    result = tuple(layers)
    if not result:
        raise ValueError("bert_pruning_config requires at least one encoder layer.")
    if not all(isinstance(layer, nn.Module) for layer in result):
        raise TypeError("BERT encoder layers must be nn.Module instances.")
    return result


def _bert_head_pruner(
    model: nn.Module,
    layer_index: int,
) -> Callable[[nn.Module, tuple[int, ...]], None]:
    def prune(
        attention_module: nn.Module,
        current_head_positions: tuple[int, ...],
    ) -> None:
        self_attention = getattr(attention_module, "self", None)
        current_heads = int(self_attention.num_attention_heads)
        original_heads = int(model.config.num_attention_heads)
        already_pruned = {
            int(head) for head in getattr(attention_module, "pruned_heads", set())
        }
        remaining_original_heads = tuple(
            head for head in range(original_heads) if head not in already_pruned
        )
        if len(remaining_original_heads) != current_heads:
            raise ValueError(
                "BERT pruned-head metadata is inconsistent with the current "
                "attention projection width."
            )

        positions = tuple(
            sorted({int(position) for position in current_head_positions})
        )
        if not positions:
            return
        if positions[0] < 0 or positions[-1] >= current_heads:
            raise IndexError("BERT current head position is outside the layer.")
        heads = [remaining_original_heads[position] for position in positions]
        model.prune_heads({int(layer_index): heads})

    return prune


def _bert_target_group_filter(
    ffn_pairs: tuple[tuple[nn.Linear, nn.Linear], ...],
) -> Callable[[CanonicalUnitGroup], bool]:
    def keep(group: CanonicalUnitGroup) -> bool:
        if group.attention_spec is not None:
            return True

        member_pairs = {(member.module, member.handler) for member in group.members}
        return any(
            {
                (intermediate, prune_linear_out_channels),
                (output, prune_linear_in_channels),
            }.issubset(member_pairs)
            for intermediate, output in ffn_pairs
        )

    return keep


def _refresh_bert_pruning_config(tracker: WeightTracker) -> None:
    attention_specs, ffn_pairs = _bert_specs_and_ffn_pairs(tracker.model)
    tracker.attention_specs = attention_specs
    tracker.canonical_group_filter = _bert_target_group_filter(ffn_pairs)
