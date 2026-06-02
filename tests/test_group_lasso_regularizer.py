from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.fixtures_models import TinyTransformerClassifier
from tests.test_calculation_specs import _model_and_groups
from torch_weighttracker.calculations import CalcType
from torch_weighttracker.canonical_units import canonicalize_groups
from torch_weighttracker.regularizers import RegularizerType
from torch_weighttracker.regularizers.group_lasso import GroupLasso
from torch_weighttracker.torch_pruning.pruner.function import (
    prune_batchnorm_out_channels,
    prune_conv_in_channels,
    prune_conv_out_channels,
    prune_linear_in_channels,
    prune_linear_out_channels,
    prune_multihead_attention_out_channels,
)
from torch_weighttracker.weight_tracker import WeightTracker


class ParameterCalculation(nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = nn.Parameter(value)
        self.forward_calls = 0

    def forward(self) -> torch.Tensor:
        self.forward_calls += 1
        return self.value


class ReusableParamCalculation(nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = nn.Parameter(value)
        self.forward_calls = 0
        self.forward_from_l2_norm_calls = 0
        self.seen_l2_norm_pr_unit = None

    def forward(self) -> torch.Tensor:
        self.forward_calls += 1
        return self.value

    def forward_from_l2_norm_pr_unit(
        self,
        l2_norm_pr_unit: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_from_l2_norm_calls += 1
        self.seen_l2_norm_pr_unit = l2_norm_pr_unit
        return self.value


class FakeGroup:
    def __init__(self, *items) -> None:
        self.items = list(items)


def _member(module: nn.Module, handler, indices: tuple[int, ...]):
    return SimpleNamespace(
        dep=SimpleNamespace(
            target=SimpleNamespace(module=module),
            handler=handler,
        ),
        root_idxs=indices,
        idxs=indices,
    )


def test_group_lasso_declares_param_unit_and_l2_dependencies() -> None:
    assert GroupLasso.required_calculations == (
        CalcType.PARAM_PR_UNIT,
        CalcType.L2_NORM_PR_UNIT,
    )


def test_group_lasso_direct_formula_and_param_and_l2_gradients() -> None:
    param_pr_unit = ParameterCalculation(torch.tensor([3.0, 4.0, 3.0, 2.0]))
    l2_norm_pr_unit = ParameterCalculation(torch.tensor([5.0, 7.0, 11.0, 13.0]))
    regularizer = GroupLasso(
        {
            CalcType.PARAM_PR_UNIT: param_pr_unit,
            CalcType.L2_NORM_PR_UNIT: l2_norm_pr_unit,
        }
    )

    loss = regularizer()
    loss.backward()

    expected_loss = (
        torch.sqrt(torch.tensor(3.0)) * 5.0
        + 2.0 * 7.0
        + torch.sqrt(torch.tensor(3.0)) * 11.0
        + torch.sqrt(torch.tensor(2.0)) * 13.0
    )
    torch.testing.assert_close(loss.detach(), expected_loss)
    torch.testing.assert_close(
        param_pr_unit.value.grad,
        torch.tensor(
            [
                5.0 / (2.0 * torch.sqrt(torch.tensor(3.0))),
                7.0 / 4.0,
                11.0 / (2.0 * torch.sqrt(torch.tensor(3.0))),
                13.0 / (2.0 * torch.sqrt(torch.tensor(2.0))),
            ]
        ),
    )
    torch.testing.assert_close(
        l2_norm_pr_unit.value.grad,
        torch.tensor(
            [
                torch.sqrt(torch.tensor(3.0)),
                2.0,
                torch.sqrt(torch.tensor(3.0)),
                torch.sqrt(torch.tensor(2.0)),
            ]
        ),
    )


def test_group_lasso_reuses_l2_norm_for_param_pr_unit_fast_path() -> None:
    param_pr_unit = ReusableParamCalculation(torch.tensor([3.0, 4.0]))
    l2_norm_pr_unit = ParameterCalculation(torch.tensor([5.0, 7.0]))
    regularizer = GroupLasso(
        {
            CalcType.PARAM_PR_UNIT: param_pr_unit,
            CalcType.L2_NORM_PR_UNIT: l2_norm_pr_unit,
        }
    )

    loss = regularizer()

    torch.testing.assert_close(
        loss.detach(),
        torch.sqrt(torch.tensor(3.0)) * 5.0 + 2.0 * 7.0,
    )
    assert l2_norm_pr_unit.forward_calls == 1
    assert param_pr_unit.forward_calls == 0
    assert param_pr_unit.forward_from_l2_norm_calls == 1
    assert param_pr_unit.seen_l2_norm_pr_unit is l2_norm_pr_unit.value


def test_group_lasso_requires_explicit_calculations() -> None:
    with pytest.raises(ValueError, match="missing required calculations"):
        GroupLasso({})


def test_group_lasso_linear_chain_exact_values_and_gradients() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, groups=groups)

    expected_l2 = torch.tensor(
        [
            torch.sqrt(torch.tensor(17.0)),
            0.0,
            7.0,
            torch.sqrt(torch.tensor(52.0)),
        ]
    )
    _assert_group_lasso_calculations(
        tracker,
        expected_unit_active_mask=torch.tensor([1.0, 0.0, 1.0, 1.0]),
        expected_active_pr_group=torch.tensor([2.0, 1.0]),
        expected_baseline_group_sizes=torch.tensor([3.0, 1.0]),
        expected_group_change_effect=torch.tensor([3.0, 3.0]),
        expected_group_sizes=torch.tensor([3, 1]),
        expected_param_pr_unit=torch.tensor([3.0, 0.0, 3.0, 2.0]),
        expected_l2_norm_pr_unit=expected_l2,
    )

    regularizer = tracker.create_regularizer(RegularizerType.GROUP_LASSO)
    loss = regularizer()
    loss.backward()

    expected_loss = torch.sqrt(torch.tensor(3.0)) * (
        torch.sqrt(torch.tensor(17.0)) + 7.0
    ) + torch.sqrt(torch.tensor(2.0)) * torch.sqrt(torch.tensor(52.0))
    torch.testing.assert_close(loss.detach(), expected_loss)
    assert torch.isfinite(model.fc1.weight.grad).all()
    assert torch.isfinite(model.fc2.weight.grad).all()
    torch.testing.assert_close(
        model.fc1.weight.grad,
        torch.tensor(
            [
                [
                    torch.sqrt(torch.tensor(3.0)) / torch.sqrt(torch.tensor(17.0)),
                    0.0,
                ],
                [0.0, 0.0],
                [
                    2.0 * torch.sqrt(torch.tensor(3.0)) / 7.0,
                    3.0 * torch.sqrt(torch.tensor(3.0)) / 7.0,
                ],
            ]
        ),
    )
    torch.testing.assert_close(
        model.fc2.weight.grad,
        torch.tensor(
            [
                [
                    4.0 * torch.sqrt(torch.tensor(3.0))
                    / torch.sqrt(torch.tensor(17.0))
                    + 4.0
                    * torch.sqrt(torch.tensor(2.0))
                    / torch.sqrt(torch.tensor(52.0)),
                    0.0,
                    6.0 * torch.sqrt(torch.tensor(3.0)) / 7.0
                    + 6.0
                    * torch.sqrt(torch.tensor(2.0))
                    / torch.sqrt(torch.tensor(52.0)),
                ]
            ]
        ),
    )


def test_group_lasso_include_filters_members_not_weighted_modules() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, groups=groups)

    context = GroupLasso.calculation_context(tracker, include=[model.fc1])
    assert context is not None
    assert context.weighted_modules == tracker._get_weighted_modules()

    regularizer = tracker.create_regularizer(
        RegularizerType.GROUP_LASSO,
        include=[model.fc1],
    )
    before = regularizer()

    with torch.no_grad():
        model.fc2.weight.add_(1000.0)

    torch.testing.assert_close(regularizer(), before)

    with torch.no_grad():
        model.fc1.weight.add_(1.0)

    assert not torch.allclose(regularizer(), before)


class ConvBnChain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.bn1(self.conv1(x)))


def test_group_lasso_conv_bn_chain_exact_values_and_gradients() -> None:
    model = ConvBnChain()
    with torch.no_grad():
        model.conv1.weight.copy_(torch.tensor([[[[2.0]]], [[[0.0]]]]))
        model.bn1.weight.copy_(torch.tensor([4.0, 0.0]))
        model.conv2.weight.copy_(torch.tensor([[[[5.0]], [[0.0]]]]))

    groups = canonicalize_groups(
        (
            FakeGroup(
                _member(model.conv1, prune_conv_out_channels, (0, 1)),
                _member(model.bn1, prune_batchnorm_out_channels, (0, 1)),
                _member(model.conv2, prune_conv_in_channels, (0, 1)),
            ),
        )
    )
    tracker = WeightTracker(model, groups=groups)

    _assert_group_lasso_calculations(
        tracker,
        expected_unit_active_mask=torch.tensor([1.0, 0.0]),
        expected_active_pr_group=torch.tensor([1.0]),
        expected_baseline_group_sizes=torch.tensor([2.0]),
        expected_group_change_effect=torch.tensor([4.0]),
        expected_group_sizes=torch.tensor([2]),
        expected_param_pr_unit=torch.tensor([3.0, 0.0]),
        expected_l2_norm_pr_unit=torch.tensor([torch.sqrt(torch.tensor(45.0)), 0.0]),
    )

    loss = tracker.create_regularizer(RegularizerType.GROUP_LASSO)()
    loss.backward()

    torch.testing.assert_close(
        loss.detach(),
        torch.sqrt(torch.tensor(3.0)) * torch.sqrt(torch.tensor(45.0)),
    )
    assert torch.isfinite(model.conv1.weight.grad).all()
    assert torch.isfinite(model.bn1.weight.grad).all()
    assert torch.isfinite(model.conv2.weight.grad).all()
    torch.testing.assert_close(
        model.conv1.weight.grad,
        torch.tensor(
            [
                [
                    [
                        [
                            2.0
                            * torch.sqrt(torch.tensor(3.0))
                            / torch.sqrt(torch.tensor(45.0))
                        ]
                    ]
                ],
                [[[0.0]]],
            ]
        ),
    )
    torch.testing.assert_close(
        model.bn1.weight.grad,
        torch.tensor(
            [
                4.0 * torch.sqrt(torch.tensor(3.0)) / torch.sqrt(torch.tensor(45.0)),
                0.0,
            ]
        ),
    )
    torch.testing.assert_close(
        model.conv2.weight.grad,
        torch.tensor(
            [
                [
                    [
                        [
                            5.0
                            * torch.sqrt(torch.tensor(3.0))
                            / torch.sqrt(torch.tensor(45.0))
                        ]
                    ],
                    [[0.0]],
                ]
            ]
        ),
    )


class TinyQKVProjectionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(4, 12, bias=False)
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qkv(x)[..., :4] + self.proj(x)


@dataclass(frozen=True)
class QKVGroupLassoCase:
    name: str
    prune_dim: bool | None
    prune_num_heads: bool
    zero_embed_indices: tuple[int, ...]
    expected_unit_active_mask: torch.Tensor
    expected_active_pr_group: torch.Tensor
    expected_baseline_group_sizes: torch.Tensor
    expected_group_change_effect: torch.Tensor
    expected_group_sizes: torch.Tensor
    expected_param_pr_unit: torch.Tensor
    expected_l2_norm_pr_unit: torch.Tensor
    expected_loss: torch.Tensor
    expected_qkv_grad_active_rows: tuple[int, ...]
    expected_proj_grad_active_cols: tuple[int, ...]
    expected_grad_value: torch.Tensor | float


QKV_GROUP_LASSO_CASES = (
    QKVGroupLassoCase(
        name="channel",
        prune_dim=False,
        prune_num_heads=False,
        zero_embed_indices=(1,),
        expected_unit_active_mask=torch.tensor([1.0, 0.0, 1.0, 1.0]),
        expected_active_pr_group=torch.tensor([3.0]),
        expected_baseline_group_sizes=torch.tensor([4.0]),
        expected_group_change_effect=torch.tensor([16.0]),
        expected_group_sizes=torch.tensor([4]),
        expected_param_pr_unit=torch.tensor([16.0, 0.0, 16.0, 16.0]),
        expected_l2_norm_pr_unit=torch.tensor([4.0, 0.0, 4.0, 4.0]),
        expected_loss=torch.tensor(48.0),
        expected_qkv_grad_active_rows=(0, 2, 3, 4, 6, 7, 8, 10, 11),
        expected_proj_grad_active_cols=(0, 2, 3),
        expected_grad_value=1.0,
    ),
    QKVGroupLassoCase(
        name="head",
        prune_dim=False,
        prune_num_heads=True,
        zero_embed_indices=(2, 3),
        expected_unit_active_mask=torch.tensor([1.0, 0.0]),
        expected_active_pr_group=torch.tensor([1.0]),
        expected_baseline_group_sizes=torch.tensor([2.0]),
        expected_group_change_effect=torch.tensor([32.0]),
        expected_group_sizes=torch.tensor([2]),
        expected_param_pr_unit=torch.tensor([32.0, 0.0]),
        expected_l2_norm_pr_unit=torch.tensor([torch.sqrt(torch.tensor(32.0)), 0.0]),
        expected_loss=torch.tensor(32.0),
        expected_qkv_grad_active_rows=(0, 1, 4, 5, 8, 9),
        expected_proj_grad_active_cols=(0, 1),
        expected_grad_value=1.0,
    ),
    QKVGroupLassoCase(
        name="head_dim",
        prune_dim=True,
        prune_num_heads=False,
        zero_embed_indices=(1, 3),
        expected_unit_active_mask=torch.tensor([1.0, 0.0]),
        expected_active_pr_group=torch.tensor([1.0]),
        expected_baseline_group_sizes=torch.tensor([2.0]),
        expected_group_change_effect=torch.tensor([32.0]),
        expected_group_sizes=torch.tensor([2]),
        expected_param_pr_unit=torch.tensor([32.0, 0.0]),
        expected_l2_norm_pr_unit=torch.tensor([torch.sqrt(torch.tensor(32.0)), 0.0]),
        expected_loss=torch.tensor(32.0),
        expected_qkv_grad_active_rows=(0, 2, 4, 6, 8, 10),
        expected_proj_grad_active_cols=(0, 2),
        expected_grad_value=1.0,
    ),
)


@pytest.mark.parametrize(
    "case",
    QKV_GROUP_LASSO_CASES,
    ids=[case.name for case in QKV_GROUP_LASSO_CASES],
)
def test_group_lasso_qkv_projection_exact_values_and_gradients(
    case: QKVGroupLassoCase,
) -> None:
    model = TinyQKVProjectionBlock()
    with torch.no_grad():
        model.qkv.weight.fill_(1.0)
        model.proj.weight.fill_(1.0)
        _zero_qkv_embed_indices(model.qkv, case.zero_embed_indices)
        model.proj.weight[:, case.zero_embed_indices] = 0.0

    groups = canonicalize_groups(
        (
            FakeGroup(
                _member(model.proj, prune_linear_in_channels, (0, 1, 2, 3)),
                _member(model.qkv, prune_linear_out_channels, tuple(range(12))),
            ),
        ),
        num_heads={model.qkv: 2},
        prune_dim=case.prune_dim,
        prune_num_heads=case.prune_num_heads,
    )
    assert tuple(group.length for group in groups) == tuple(
        int(size) for size in case.expected_group_sizes
    )
    tracker = WeightTracker(
        model,
        groups=groups,
        num_heads={model.qkv: 2},
        prune_dim=case.prune_dim,
        prune_num_heads=case.prune_num_heads,
    )

    _assert_group_lasso_calculations(
        tracker,
        expected_unit_active_mask=case.expected_unit_active_mask,
        expected_active_pr_group=case.expected_active_pr_group,
        expected_baseline_group_sizes=case.expected_baseline_group_sizes,
        expected_group_change_effect=case.expected_group_change_effect,
        expected_group_sizes=case.expected_group_sizes,
        expected_param_pr_unit=case.expected_param_pr_unit,
        expected_l2_norm_pr_unit=case.expected_l2_norm_pr_unit,
    )

    loss = tracker.create_regularizer(RegularizerType.GROUP_LASSO)()
    loss.backward()

    torch.testing.assert_close(loss.detach(), case.expected_loss)
    assert torch.isfinite(model.qkv.weight.grad).all()
    assert torch.isfinite(model.proj.weight.grad).all()
    torch.testing.assert_close(
        model.qkv.weight.grad,
        _expected_qkv_grad(
            active_rows=case.expected_qkv_grad_active_rows,
            value=case.expected_grad_value,
        ),
    )
    torch.testing.assert_close(
        model.proj.weight.grad,
        _expected_proj_grad(
            active_cols=case.expected_proj_grad_active_cols,
            value=case.expected_grad_value,
        ),
    )


def test_group_lasso_multihead_attention_reaches_out_projection_weights() -> None:
    attention = nn.MultiheadAttention(
        embed_dim=4,
        num_heads=2,
        batch_first=True,
        bias=False,
    )
    model = nn.Module()
    model.attn = attention
    groups = canonicalize_groups(
        (
            FakeGroup(
                _member(
                    attention,
                    prune_multihead_attention_out_channels,
                    (0, 1, 2, 3),
                ),
            ),
        ),
        num_heads={attention: attention.num_heads},
        prune_num_heads=True,
    )
    tracker = WeightTracker(
        model,
        groups=groups,
        num_heads={attention: attention.num_heads},
        prune_num_heads=True,
    )

    loss = tracker.create_regularizer(RegularizerType.GROUP_LASSO)()
    loss.backward()

    assert attention.in_proj_weight.grad is not None
    assert attention.out_proj.weight.grad is not None
    assert attention.in_proj_weight.grad.abs().sum() > 0
    assert attention.out_proj.weight.grad.abs().sum() > 0


def test_group_lasso_terms_are_zero_for_fake_zeroed_transformer_structures() -> None:
    torch.manual_seed(0)
    model = TinyTransformerClassifier().eval()
    token_ids = torch.randint(0, 32, (1, 8))
    tracker = WeightTracker(
        model,
        example_inputs=token_ids,
        root_module_types=[nn.MultiheadAttention, nn.Linear],
        num_heads={model.attn: model.attn.num_heads},
        prune_num_heads=True,
    )

    attention_group = _group_containing(
        tracker,
        "attn",
        prune_multihead_attention_out_channels,
    )
    mlp_group = _group_containing(
        tracker,
        "mlp_in",
        prune_linear_out_channels,
    )
    _zero_group_unit(attention_group, (0, 1, 2, 3))
    _zero_group_unit(mlp_group, (0, 7, 31))

    l2_norm_pr_unit = tracker.get_calculation(CalcType.L2_NORM_PR_UNIT)()
    param_pr_unit = tracker.get_calculation(CalcType.PARAM_PR_UNIT)()
    group_lasso_terms = param_pr_unit.sqrt() * l2_norm_pr_unit
    loss = tracker.create_regularizer(RegularizerType.GROUP_LASSO)()
    zeroed_unit_indices = torch.tensor(
        [
            attention_group.offset,
            mlp_group.offset,
            mlp_group.offset + 7,
            mlp_group.offset + 31,
        ]
    )

    torch.testing.assert_close(
        l2_norm_pr_unit.index_select(0, zeroed_unit_indices),
        torch.zeros(4),
    )
    torch.testing.assert_close(
        param_pr_unit.index_select(0, zeroed_unit_indices),
        torch.zeros(4),
    )
    torch.testing.assert_close(
        group_lasso_terms.index_select(0, zeroed_unit_indices),
        torch.zeros(4),
    )
    torch.testing.assert_close(loss.detach(), group_lasso_terms.sum().detach())
    assert group_lasso_terms.sum() > 0


def _assert_group_lasso_calculations(
    tracker: WeightTracker,
    *,
    expected_unit_active_mask: torch.Tensor,
    expected_active_pr_group: torch.Tensor,
    expected_baseline_group_sizes: torch.Tensor,
    expected_group_change_effect: torch.Tensor,
    expected_group_sizes: torch.Tensor,
    expected_param_pr_unit: torch.Tensor,
    expected_l2_norm_pr_unit: torch.Tensor,
) -> None:
    unit_active_mask = tracker.get_calculation(CalcType.UNIT_ACTIVE_MASK)()
    active_pr_group = tracker.get_calculation(CalcType.UNITS_TO_GROUP)(
        unit_active_mask
    )
    baseline_group_sizes = tracker.get_calculation(CalcType.INIT_UNIT_PR_GROUP_COUNT)()
    group_change_effect = tracker.get_calculation(CalcType.GROUP_CHANGE_EFFECT)()
    group_sizes = tracker.get_calculation(CalcType.GROUP_SIZES)()
    param_pr_unit = tracker.get_calculation(CalcType.PARAM_PR_UNIT)()
    l2_norm_pr_unit = tracker.get_calculation(CalcType.L2_NORM_PR_UNIT)()

    torch.testing.assert_close(unit_active_mask, expected_unit_active_mask)
    torch.testing.assert_close(active_pr_group, expected_active_pr_group)
    torch.testing.assert_close(baseline_group_sizes, expected_baseline_group_sizes)
    torch.testing.assert_close(group_change_effect, expected_group_change_effect)
    torch.testing.assert_close(group_sizes, expected_group_sizes)
    torch.testing.assert_close(param_pr_unit, expected_param_pr_unit)
    torch.testing.assert_close(l2_norm_pr_unit, expected_l2_norm_pr_unit)


def _group_containing(
    tracker: WeightTracker,
    module_name: str,
    handler,
):
    names = {module: name for name, module in tracker.model.named_modules()}
    for group in tracker.canonical_groups:
        for member in group.members:
            if names[member.module] == module_name and member.handler == handler:
                return group
    raise AssertionError(f"No canonical group contains {module_name}.")


def _zero_group_unit(
    group,
    root_indices: tuple[int, ...],
) -> None:
    roots = set(root_indices)
    for member in group.members:
        raw_member = member.member
        local_indices = [
            int(local_index)
            for local_index, root_index in zip(
                raw_member.idxs,
                raw_member.root_idxs,
                strict=True,
            )
            if int(root_index) in roots
        ]
        if local_indices:
            _zero_member_slices(member.module, member.handler, local_indices)


def _zero_member_slices(
    module: nn.Module,
    handler,
    indices: list[int],
) -> None:
    with torch.no_grad():
        if isinstance(module, nn.Linear) and handler == prune_linear_out_channels:
            module.weight[indices, :] = 0
            if module.bias is not None:
                module.bias[indices] = 0
            return

        if isinstance(module, nn.Linear) and handler == prune_linear_in_channels:
            module.weight[:, indices] = 0
            return

        if isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                module.weight[indices] = 0
                if module.bias is not None:
                    module.bias[indices] = 0
            return

        if (
            isinstance(module, nn.MultiheadAttention)
            and handler == prune_multihead_attention_out_channels
        ):
            _zero_mha_out_slices(module, indices)
            return

    raise AssertionError(f"Unhandled test zeroing rule for {handler.__name__}")


def _zero_mha_out_slices(attention: nn.MultiheadAttention, indices: list[int]) -> None:
    embed_dim = attention.embed_dim
    repeated = list(indices)
    repeated += [index + embed_dim for index in indices]
    repeated += [index + 2 * embed_dim for index in indices]

    if attention.in_proj_weight is not None:
        attention.in_proj_weight[repeated, :] = 0
        attention.in_proj_weight[:, indices] = 0
    if attention.in_proj_bias is not None:
        attention.in_proj_bias[repeated] = 0
    if attention.out_proj is not None:
        attention.out_proj.weight[indices, :] = 0
        attention.out_proj.weight[:, indices] = 0
        if attention.out_proj.bias is not None:
            attention.out_proj.bias[indices] = 0


def _zero_qkv_embed_indices(module: nn.Linear, embed_indices: tuple[int, ...]) -> None:
    embed_dim = module.out_features // 3
    row_indices = []
    for base in (0, embed_dim, 2 * embed_dim):
        row_indices.extend(base + index for index in embed_indices)
    module.weight[row_indices, :] = 0.0


def _expected_qkv_grad(
    *,
    active_rows: tuple[int, ...],
    value: float,
) -> torch.Tensor:
    grad = torch.zeros(12, 4)
    grad[list(active_rows), :] = value
    return grad


def _expected_proj_grad(
    *,
    active_cols: tuple[int, ...],
    value: float,
) -> torch.Tensor:
    grad = torch.zeros(4, 4)
    grad[:, list(active_cols)] = value
    return grad
