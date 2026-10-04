from __future__ import annotations

import math
import warnings

import pytest
import torch
import torch.nn as nn

from tests.fixtures_models import CifarResNet20
from torch_weighttracker import (
    ReductionBackend,
    SparseReductionFallbackWarning,
    WeightTracker,
    use_reduction_backend,
)
from torch_weighttracker.calculations import CalcType, PipelineCalc, ReductionCalc
from torch_weighttracker.extractors.extractor import (
    ModuleParameterRef,
    TensorSpec,
    ValueTensorRef,
)
from torch_weighttracker.operations import generic
from torch_weighttracker.reductions import sparse
from torch_weighttracker.reductions.builder import (
    FullSelection,
    IndexSelection,
    ReductionMapping,
    ReductionPlanBuilder,
    ReductionRecord,
    SegmentSelection,
)
from torch_weighttracker.reductions.ops import IdentityTensorReduction, ReductionOp

ALL_TRACKERS = (
    "structured_bops",
    "unstructured_bops",
    "unstructured_sparsity",
    "group_pruning_summary",
    "l2_norm_distribution",
    "nvidia_2_4_sparsity",
)

PLAN_CALCS = (
    CalcType.ACTIVE_UNITS,
    CalcType.L2_NORM_PR_UNIT,
    CalcType.UNIT_ACTIVE_MASK,
    CalcType.PARAM_PR_UNIT,
    CalcType.ACTIVE_MACS_PR_MODULE,
    CalcType.UNSTRUCTURED_SPARSITY_PR_MODULE,
)


def _sparsify_(model: nn.Module, seed: int = 0) -> None:
    """Zero random elements and whole output channels of conv/linear weights."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                keep = torch.rand(weight.shape, generator=generator) > 0.3
                weight.mul_(keep.to(weight))
                dead = torch.rand(weight.shape[0], generator=generator) < 0.25
                weight[dead] = 0


def _resnet20_pair():
    model = CifarResNet20().eval()
    _sparsify_(model)
    example = torch.randn(1, 3, 32, 32)
    sparse_tracker = WeightTracker(model, example_inputs=example)
    loop_tracker = WeightTracker(
        model, example_inputs=example, reduction_backend="loop"
    )
    return model, sparse_tracker, loop_tracker


def _assert_metrics_close(actual, expected, path: str = "") -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_metrics_close(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_metrics_close(a, e, f"{path}[{index}]")
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6), (
            path,
            actual,
            expected,
        )
    else:
        assert actual == expected, path


def _plan_calcs(tracker: WeightTracker):
    found = {}
    for calculation in tracker.calculations.values():
        for module in calculation.modules():
            if isinstance(module, (ReductionCalc, PipelineCalc)):
                found[id(module)] = module
    return tuple(found.values())


def test_sparse_backend_is_default_and_lowers_every_resnet_entry() -> None:
    _, tracker, loop_tracker = _resnet20_pair()
    with warnings.catch_warnings():
        warnings.simplefilter("error", SparseReductionFallbackWarning)
        tracker.create_tracker(list(ALL_TRACKERS))
        tracker.create_regularizer("group_lasso")
    loop_tracker.create_tracker(list(ALL_TRACKERS))

    calcs = _plan_calcs(tracker)
    assert tracker.reduction_backend is ReductionBackend.SPARSE
    assert calcs
    for calc in calcs:
        assert calc.backend is ReductionBackend.SPARSE
        assert all(calc._sparse_flags), calc.calculation_type
    assert all(
        calc.backend is ReductionBackend.LOOP for calc in _plan_calcs(loop_tracker)
    )


def test_all_tracker_metrics_match_loop_backend_on_sparsified_resnet20() -> None:
    _, tracker, loop_tracker = _resnet20_pair()
    for owner in (tracker, loop_tracker):
        owner.create_tracker(list(ALL_TRACKERS))
        owner.create_tracker(
            "structured_bops",
            log_total_bops=True,
            log_layerwise_stats=True,
            wandb_format=True,
        )

    _assert_metrics_close(tracker.track(), loop_tracker.track())


@pytest.mark.parametrize("calc_type", PLAN_CALCS, ids=lambda c: c.value)
def test_calculation_outputs_match_loop_backend(calc_type: CalcType) -> None:
    _, tracker, loop_tracker = _resnet20_pair()

    torch.testing.assert_close(
        tracker.get_calculation(calc_type)(),
        loop_tracker.get_calculation(calc_type)(),
        rtol=1e-5,
        atol=1e-6,
    )


def test_group_lasso_loss_and_gradients_match_loop_backend() -> None:
    model, tracker, loop_tracker = _resnet20_pair()
    params = [p for p in model.parameters() if p.requires_grad]

    def loss_and_grads(owner: WeightTracker):
        for param in params:
            param.grad = None
        loss = owner.create_regularizer("group_lasso")()
        loss.backward()
        return loss.detach(), [
            torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in params
        ]

    sparse_loss, sparse_grads = loss_and_grads(tracker)
    loop_loss, loop_grads = loss_and_grads(loop_tracker)

    torch.testing.assert_close(sparse_loss, loop_loss)
    for sparse_grad, loop_grad in zip(sparse_grads, loop_grads, strict=True):
        torch.testing.assert_close(sparse_grad, loop_grad, rtol=1e-5, atol=1e-7)


def test_sparse_calculation_follows_in_place_weight_updates() -> None:
    model, tracker, loop_tracker = _resnet20_pair()
    active = tracker.get_calculation(CalcType.ACTIVE_UNITS)
    loop_active = loop_tracker.get_calculation(CalcType.ACTIVE_UNITS)
    before = active().clone()

    with torch.no_grad():
        model.blocks[0].conv1.weight[:5] = 0

    assert not torch.equal(active(), before)
    torch.testing.assert_close(active(), loop_active())


def test_sparse_backend_matches_after_physical_prune_rebuild() -> None:
    model, tracker, _ = _resnet20_pair()
    tracker.prune_zero_units()
    loop_tracker = WeightTracker(
        model,
        example_inputs=torch.randn(1, 3, 32, 32),
        reduction_backend="loop",
    )

    for calc_type in (CalcType.L2_NORM_PR_UNIT, CalcType.ACTIVE_MACS_PR_MODULE):
        torch.testing.assert_close(
            tracker.get_calculation(calc_type)(),
            loop_tracker.get_calculation(calc_type)(),
            rtol=1e-5,
            atol=1e-6,
        )


def test_executor_rejects_sources_that_changed_size() -> None:
    model, tracker, _ = _resnet20_pair()
    l2 = tracker.get_calculation(CalcType.L2_NORM_PR_UNIT)

    model.stem.weight = nn.Parameter(torch.randn(8, 3, 3, 3))

    with pytest.raises(RuntimeError, match="changed size"):
        l2()


def test_use_reduction_backend_scopes_plain_calculations() -> None:
    linear = nn.Linear(3, 2)

    with use_reduction_backend("loop"):
        loop_calc = ReductionCalc(_plan(linear, generic.SquaredSumWeight(dim=1)))
    sparse_calc = ReductionCalc(_plan(linear, generic.SquaredSumWeight(dim=1)))

    assert loop_calc.backend is ReductionBackend.LOOP
    assert sparse_calc.backend is ReductionBackend.SPARSE
    torch.testing.assert_close(sparse_calc(), loop_calc())


# ---------------------------------------------------------------- operation forms


def _plan(module: nn.Module, reduction, *, target=None, source=None):
    builder = ReductionPlanBuilder()
    op = ReductionOp(ModuleParameterRef(module, "weight"), reduction)
    builder.add(
        ReductionRecord(
            op=op,
            mapping=ReductionMapping(
                source=FullSelection() if source is None else source,
                target=FullSelection() if target is None else target,
            ),
        )
    )
    return builder.finalize()


def _conv_with_zeros() -> nn.Conv2d:
    torch.manual_seed(3)
    conv = nn.Conv2d(4, 5, 3, bias=False)
    with torch.no_grad():
        conv.weight[1] = 0
        conv.weight[:, 2] = 0
        conv.weight[3, 0, 1] = 0
    return conv


OPERATIONS = (
    generic.SumWeight(dim=(1, 2, 3)),
    generic.SquaredSumWeight(dim=(0, 2, 3)),
    generic.MeanWeight(dim=(1, 2, 3)),
    generic.CountWeight(dim=(0, 2, 3)),
    generic.ActiveWeight(dim=(1, 2, 3)),
    generic.ActiveWeight(dim=(0, 2, 3)),
    generic.ActiveWeight(dim=None),
    generic.ActiveWeight(dim=()),
    generic.L1Weight(dim=(1, 2, 3)),
    generic.L2Weight(dim=(0, 2, 3)),
    generic.L2Weight(dim=-1, keepdim=True),
    generic.ElementwiseSumWeight(dim=()),
    generic.ElementwiseSquaredSumWeight(dim=()),
    generic.ElementwiseL2Weight(dim=()),
)


@pytest.mark.parametrize("reduction", OPERATIONS, ids=lambda op: repr(op))
def test_every_generic_operation_lowers_exactly(reduction) -> None:
    conv = _conv_with_zeros()
    plan = _plan(conv, reduction)

    with use_reduction_backend("loop"):
        expected = ReductionCalc(plan)()
    calc = ReductionCalc(plan)

    assert calc.backend is ReductionBackend.SPARSE
    torch.testing.assert_close(calc(), expected)


def test_l2_lowering_has_zero_gradient_for_dead_slices() -> None:
    conv = _conv_with_zeros()
    plan = _plan(conv, generic.L2Weight(dim=(1, 2, 3)))

    with use_reduction_backend("loop"):
        ReductionCalc(plan)().sum().backward()
    expected = conv.weight.grad.clone()
    conv.weight.grad = None
    ReductionCalc(plan)().sum().backward()

    torch.testing.assert_close(conv.weight.grad, expected)
    assert torch.isfinite(conv.weight.grad).all()
    assert conv.weight.grad[1].abs().sum() == 0


def test_indexed_and_gather_targets_lower_with_duplicate_destinations() -> None:
    conv = _conv_with_zeros()
    builder = ReductionPlanBuilder(output_length=4)
    for reduction, source, target in (
        (
            generic.SquaredSumWeight(dim=(1, 2, 3)),
            None,
            IndexSelection((3, 3, 0, 1, 2)),
        ),
        (
            generic.ActiveWeight(dim=(0, 2, 3)),
            IndexSelection((0, 2, 2)),
            IndexSelection((1, 1, 3)),
        ),
        (
            generic.SumWeight(dim=(1, 2, 3)),
            SegmentSelection(1, 3),
            SegmentSelection(0, 3),
        ),
    ):
        builder.add(
            ReductionRecord(
                op=ReductionOp(ModuleParameterRef(conv, "weight"), reduction),
                mapping=ReductionMapping(
                    source=FullSelection() if source is None else source,
                    target=target,
                ),
            )
        )
    plan = builder.finalize()

    with use_reduction_backend("loop"):
        expected = ReductionCalc(plan)()
    calc = ReductionCalc(plan)

    assert calc.backend is ReductionBackend.SPARSE
    assert all(calc._sparse_flags)
    torch.testing.assert_close(calc(), expected)


class _OpaqueReduction:
    """A reduction the sparse backend cannot describe."""

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(value.shape[0], -1).amax(dim=1)

    def output_spec(self, source_spec: TensorSpec) -> TensorSpec:
        return TensorSpec(
            shape=torch.Size([source_spec.shape[0]]),
            dtype=source_spec.dtype,
            device=source_spec.device,
        )

    def identity_key(self):
        return ("opaque",)


def test_unsupported_ops_stay_on_loop_path_inside_a_sparse_plan() -> None:
    conv = _conv_with_zeros()
    builder = ReductionPlanBuilder(output_length=5)
    for reduction in (generic.SquaredSumWeight(dim=(1, 2, 3)), _OpaqueReduction()):
        builder.add(
            ReductionRecord(
                op=ReductionOp(ModuleParameterRef(conv, "weight"), reduction),
                mapping=ReductionMapping(
                    source=FullSelection(), target=FullSelection()
                ),
            )
        )
    plan = builder.finalize()

    with use_reduction_backend("loop"):
        expected = ReductionCalc(plan)()
    calc = ReductionCalc(plan)

    assert calc.backend is ReductionBackend.SPARSE
    assert calc._sparse_flags == (True, False)
    torch.testing.assert_close(calc(), expected)


def test_unsupported_dtype_uses_loop_backend_without_warning() -> None:
    linear = nn.Linear(3, 2).to(torch.float16)

    with warnings.catch_warnings():
        warnings.simplefilter("error", SparseReductionFallbackWarning)
        calc = ReductionCalc(_plan(linear, generic.SquaredSumWeight(dim=1)))

    assert calc.backend is ReductionBackend.LOOP


def test_verification_failure_warns_and_falls_back(monkeypatch) -> None:
    conv = _conv_with_zeros()
    plan = _plan(conv, generic.SquaredSumWeight(dim=(1, 2, 3)))
    monkeypatch.setitem(sparse._ELEMENTWISE, "square", torch.abs)

    with pytest.warns(SparseReductionFallbackWarning, match="differs"):
        calc = ReductionCalc(plan)

    assert calc.backend is ReductionBackend.LOOP
    torch.testing.assert_close(calc(), conv.weight.detach().square().sum(dim=(1, 2, 3)))


def test_pipeline_affine_reductions_lower_with_bias() -> None:
    from torch_weighttracker.calculations.calcs.unit_delta_to_module_axis import (
        ActiveUnitAxisDeltaReduction,
    )

    value = torch.tensor([1.0, 0.0, 1.0, 1.0])
    spec = TensorSpec(shape=value.shape, dtype=value.dtype, device=value.device)
    input_ref = ValueTensorRef(value=value, spec=spec)
    builder = ReductionPlanBuilder(output_length=3)
    for reduction, source, target in (
        (
            ActiveUnitAxisDeltaReduction(2.0),
            IndexSelection((0, 1)),
            IndexSelection((0, 0)),
        ),
        (IdentityTensorReduction(), IndexSelection((2, 3)), IndexSelection((1, 2))),
    ):
        builder.add(
            ReductionRecord(
                op=ReductionOp(input_ref, reduction),
                mapping=ReductionMapping(source=source, target=target),
            )
        )
    plan = builder.finalize(spec)

    with use_reduction_backend("loop"):
        loop_calc = PipelineCalc(plan)
    calc = PipelineCalc(plan)

    assert calc.backend is ReductionBackend.SPARSE
    for probe in (value, torch.tensor([0.0, 0.0, 3.0, -1.0])):
        torch.testing.assert_close(calc(probe), loop_calc(probe))


def test_bert_metrics_and_group_lasso_match_loop_backend() -> None:
    pytest.importorskip("transformers")
    from tests.test_bert_integration import _tiny_bert
    from torch_weighttracker.integrations.transformers import bert_pruning_config

    model, inputs = _tiny_bert()
    _sparsify_(model, seed=1)
    owners = [
        WeightTracker(
            model,
            example_inputs=inputs,
            reduction_backend=backend,
            **bert_pruning_config(model),
        )
        for backend in ("sparse", "loop")
    ]
    for owner in owners:
        owner.create_tracker(["group_pruning_summary", "l2_norm_distribution"])

    _assert_metrics_close(owners[0].track(), owners[1].track())
    torch.testing.assert_close(
        owners[0].create_regularizer("group_lasso")(),
        owners[1].create_regularizer("group_lasso")(),
    )


def test_large_sources_stay_on_loop_path(monkeypatch) -> None:
    small = nn.Linear(4, 3, bias=False)
    large = nn.Linear(8, 3, bias=False)
    monkeypatch.setitem(sparse.MAX_LOWERED_SOURCE_ELEMENTS, "cpu", 12)
    builder = ReductionPlanBuilder(output_length=3)
    for module in (small, large):
        builder.add(
            ReductionRecord(
                op=ReductionOp(
                    ModuleParameterRef(module, "weight"),
                    generic.SquaredSumWeight(dim=1),
                ),
                mapping=ReductionMapping(
                    source=FullSelection(), target=FullSelection()
                ),
            )
        )
    plan = builder.finalize()

    calc = ReductionCalc(plan)

    assert calc._sparse_flags == (True, False)
    expected = small.weight.square().sum(1) + large.weight.square().sum(1)
    torch.testing.assert_close(calc(), expected)
