from __future__ import annotations

import torch
import torch.nn as nn


class ActiveParamsPrUnit(nn.Module):
    def __init__(
        self,
        *,
        unit_to_group_acc: nn.Module,
        unit_active_mask: nn.Module,
        baseline_group_size: torch.Tensor,
        group_change_effect: torch.Tensor,
        group_lengths: torch.Tensor,
    ) -> None:
        super().__init__()
        self.unit_to_group_acc = unit_to_group_acc
        self.unit_active_mask = unit_active_mask
        self.register_buffer(
            "baseline_group_size",
            baseline_group_size.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "group_change_effect",
            group_change_effect.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "group_lengths",
            group_lengths.detach().clone().to(dtype=torch.long),
            persistent=False,
        )

    def forward(self) -> torch.Tensor:
        unit_active_mask = self.unit_active_mask()
        active_pr_group = self.unit_to_group_acc(unit_active_mask)
        group_effect = (
            active_pr_group - self.baseline_group_size
        ) * self.group_change_effect
        return torch.repeat_interleave(group_effect, self.group_lengths)
