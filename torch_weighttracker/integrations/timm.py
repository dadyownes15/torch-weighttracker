from __future__ import annotations

import torch.nn as nn

from torch_weighttracker.weight_tracker import WeightTracker


def infer_vit_num_heads(model: nn.Module) -> dict[nn.Module, int]:
    """Infer timm ViT head counts keyed by each Attention.qkv projection."""
    num_heads: dict[nn.Module, int] = {}
    for module in model.modules():
        qkv = getattr(module, "qkv", None)
        heads = getattr(module, "num_heads", None)
        if isinstance(qkv, nn.Linear) and heads is not None:
            num_heads[qkv] = int(heads)
    return num_heads


def sync_vit_attention_metadata(tracker: WeightTracker) -> None:
    """Sync timm Attention metadata after qkv physical pruning."""
    for module in tracker.model.modules():
        qkv = getattr(module, "qkv", None)
        if not isinstance(qkv, nn.Linear) or qkv not in tracker.num_heads:
            continue

        num_heads = int(tracker.num_heads[qkv])
        attn_dim = int(qkv.out_features // 3)
        if num_heads <= 0 or attn_dim % num_heads != 0:
            raise ValueError(
                "Pruned timm Attention qkv output dimension must be divisible by "
                "the tracked num_heads."
            )

        head_dim = attn_dim // num_heads
        module.num_heads = num_heads
        module.attn_dim = attn_dim
        module.head_dim = head_dim
        module.scale = head_dim**-0.5


def vit_pruning_config(model: nn.Module) -> dict:
    return {
        "root_module_types": [nn.Linear],
        "num_heads": infer_vit_num_heads(model),
        "post_prune_hooks": (sync_vit_attention_metadata,),
    }

