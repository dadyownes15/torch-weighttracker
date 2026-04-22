from __future__ import annotations

import torch.nn as nn
from copy import deepcopy

from .analysis import (
    AnalyzerConfig,
    PruneZeroResult,
    ReferenceSnapshot,
    StructureAnalyzer,
    StructureAxis,
    StructureInspectionReport,
)
from .regularizers import CoupledGroupLasso
from .utils import format_structure_summary


class SparsityTracker:
    def __init__(
        self,
        model: nn.Module,
        example_inputs,
        *,
        forward_fn=None,
        output_transform=None,
        root_module_types=(),
        ignored_layers=(),
        ignored_params=(),
        customized_pruners=None,
        unwrapped_parameters=None,
        num_heads=None,
        in_channel_groups=None,
        out_channel_groups=None,
        prune_num_heads: bool = False,
        prune_head_dims: bool = True,
        verbose: bool = True,
    ):
        self.model = model
        self.config = AnalyzerConfig(
            example_inputs=example_inputs,
            forward_fn=forward_fn,
            output_transform=output_transform,
            root_module_types=tuple(root_module_types),
            ignored_layers=tuple(ignored_layers),
            ignored_params=tuple(ignored_params),
            customized_pruners=customized_pruners,
            unwrapped_parameters=unwrapped_parameters,
            num_heads=num_heads,
            in_channel_groups=in_channel_groups,
            out_channel_groups=out_channel_groups,
            prune_num_heads=prune_num_heads,
            prune_head_dims=prune_head_dims,
            verbose=verbose,
        )
        self.analyzer = StructureAnalyzer(model, self.config)

    @property
    def dg(self):
        return self.analyzer.dg

    def rebuild(self) -> None:
        self.analyzer.rebuild()

    def iter_groups(self):
        return self.analyzer.iter_groups()

    def capture_reference(self) -> ReferenceSnapshot:
        return self.analyzer.capture_reference()

    def structured_sparsity(self, tol: float = 0.0, reference: ReferenceSnapshot | None = None, include_bias: bool = False):
        return self.analyzer.structured_sparsity(
            tol=tol,
            reference=reference,
            include_bias=include_bias,
        )

    def list_structures(self, include_bias: bool = False) -> StructureInspectionReport:
        return self.analyzer.inspect_structures(include_bias=include_bias)

    def format_structure_summary(self, include_bias: bool = False) -> str:
        report = self.list_structures(include_bias=include_bias)
        return format_structure_summary(report)

    def unstructured_sparsity(self, tol: float = 0.0, only_prunable: bool = False):
        return self.analyzer.unstructured_sparsity(
            tol=tol,
            only_prunable=only_prunable,
        )

    def group_lasso(
        self,
        *,
        eps: float = 1e-8,
        include_bias: bool = False,
        weighting: str = "sqrt_size",
        reduction: str = "sum",
        axes=None,
    ):
        regularizer = CoupledGroupLasso(
            analyzer=self.analyzer,
            eps=eps,
            include_bias=include_bias,
            weighting=weighting,
            reduction=reduction,
            axes=axes,
        )
        total, named_terms = regularizer()
        return total, named_terms

    def zero_structure_candidates(self, tol: float = 0.0, include_bias: bool = False):
        self.rebuild()
        return self.analyzer.zero_structure_candidates(
            tol=tol,
            include_bias=include_bias,
        )

    def prune_zero_structures(
        self,
        tol: float = 0.0,
        include_bias: bool = False,
        max_steps: int | None = None,
    ) -> PruneZeroResult:
        pruned_group_ids = []
        final_candidates = ()
        step = 0

        while True:
            self.rebuild()
            candidates = self.analyzer.zero_structure_candidates(
                tol=tol,
                include_bias=include_bias,
            )
            final_candidates = candidates
            if len(candidates) == 0:
                break
            if max_steps is not None and step >= max_steps:
                break

            pruned_any = False
            for candidate in candidates:
                pruning_group = self.dg.get_pruning_group(
                    candidate.root_module,
                    candidate.root_handler,
                    list(candidate.root_indices),
                )
                if not self.dg.check_pruning_group(pruning_group):
                    continue
                pruning_group.prune()
                if candidate.axis == StructureAxis.HEAD:
                    self._update_num_heads_after_head_prune(candidate)
                pruned_group_ids.append(candidate.group_id)
                pruned_any = True
                step += 1
                break

            if not pruned_any:
                break

        self.rebuild()
        return PruneZeroResult(
            candidates=tuple(final_candidates),
            pruned_group_ids=tuple(pruned_group_ids),
        )

    def count_macs_naive(self):
        self_copy = deepcopy(self)

        result = self_copy.prune_zero_structures()
        
        


        

    def _update_num_heads_after_head_prune(self, candidate) -> None:
        pruned_heads = len(candidate.zero_prune_units)
        if pruned_heads == 0:
            return
        updated_num_heads = dict(self.config.num_heads or {})
        for module in candidate.attention_modules:
            if module not in updated_num_heads:
                continue
            updated_num_heads[module] = max(updated_num_heads[module] - pruned_heads, 1)
        self.config = AnalyzerConfig(
            example_inputs=self.config.example_inputs,
            forward_fn=self.config.forward_fn,
            output_transform=self.config.output_transform,
            root_module_types=self.config.root_module_types,
            ignored_layers=self.config.ignored_layers,
            ignored_params=self.config.ignored_params,
            customized_pruners=self.config.customized_pruners,
            unwrapped_parameters=self.config.unwrapped_parameters,
            num_heads=updated_num_heads,
            in_channel_groups=self.config.in_channel_groups,
            out_channel_groups=self.config.out_channel_groups,
            prune_num_heads=self.config.prune_num_heads,
            prune_head_dims=self.config.prune_head_dims,
            verbose=self.config.verbose,
        )
        self.analyzer.config = self.config
