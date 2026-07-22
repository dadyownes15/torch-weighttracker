from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from torch_weighttracker import SeparateQKVAttentionSpec, WeightTracker
from torch_weighttracker.calculations import CalcType
from torch_weighttracker.canonical_units import UnitKind, canonicalize_groups
from torch_weighttracker.integrations.transformers import bert_pruning_config
from torch_weighttracker.torch_pruning.pruner.function import (
    prune_linear_in_channels,
    prune_linear_out_channels,
)

transformers = pytest.importorskip("transformers")
BertConfig = transformers.BertConfig
BertForQuestionAnswering = transformers.BertForQuestionAnswering


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


class FakeW4(nn.Module):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        scale = weight.detach().abs().max().clamp_min(1e-8) / 7.0
        return (weight / scale).round().clamp(-8, 7) * scale


def _tiny_bert(device: str | torch.device = "cpu"):
    config = BertConfig(
        vocab_size=101,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=24,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    torch.manual_seed(7)
    model = BertForQuestionAnswering(config).to(device).eval()
    inputs = (
        torch.tensor([[2, 5, 7, 11, 13, 17, 19, 23]], device=device),
        torch.ones(1, 8, dtype=torch.long, device=device),
        torch.zeros(1, 8, dtype=torch.long, device=device),
    )
    return model, inputs


def _tracker(model: nn.Module, inputs) -> WeightTracker:
    return WeightTracker(
        model,
        example_inputs=inputs,
        **bert_pruning_config(model),
    )


def _group_signature(tracker: WeightTracker) -> tuple:
    names = {module: name for name, module in tracker.model.named_modules()}
    return tuple(
        (
            group.length,
            group.unit_kind,
            tuple(
                (
                    names[member.module],
                    member.unit_axis,
                    tuple(member.destination.indices),
                )
                for member in group.members
            ),
        )
        for group in tracker.canonical_groups
    )


def _target_modules(tracker: WeightTracker) -> tuple[nn.Linear, ...]:
    modules: list[nn.Linear] = []
    for group in tracker.canonical_groups:
        for member in group.members:
            if isinstance(member.module, nn.Linear) and member.module not in modules:
                modules.append(member.module)
    return tuple(modules)


def _fill_target_weights(tracker: WeightTracker, value: float) -> None:
    with torch.no_grad():
        for module in _target_modules(tracker):
            module.weight.fill_(value)


def _num_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_separate_qkv_spec_replaces_raw_group_and_filter_reindexes() -> None:
    query = nn.Linear(8, 8)
    key = nn.Linear(8, 8)
    value = nn.Linear(8, 8)
    output = nn.Linear(8, 8)
    owner = nn.Module()
    owner.query = query
    owner.key = key
    owner.value = value
    owner.output = output
    extra = nn.Linear(4, 3)
    spec = SeparateQKVAttentionSpec(
        query_projection=query,
        key_projection=key,
        value_projection=value,
        output_projection=output,
        attention_module=owner,
        num_heads=2,
        prune_heads=lambda module, positions: None,
    )
    groups = canonicalize_groups(
        (
            FakeGroup(
                _member(extra, prune_linear_out_channels, tuple(range(3))),
            ),
            FakeGroup(
                _member(query, prune_linear_out_channels, tuple(range(8))),
                _member(key, prune_linear_out_channels, tuple(range(8))),
                _member(value, prune_linear_out_channels, tuple(range(8))),
                _member(output, prune_linear_in_channels, tuple(range(8))),
            ),
        ),
        attention_specs=(spec,),
        canonical_group_filter=lambda group: group.attention_spec is not None,
    )

    assert len(groups) == 1
    assert groups[0].group_id == 0
    assert groups[0].offset == 0
    assert groups[0].length == 2
    assert groups[0].unit_kind == UnitKind.HEAD
    assert all(
        tuple(member.destination.indices) == (0, 0, 0, 0, 1, 1, 1, 1)
        for member in groups[0].members
    )


def test_separate_qkv_spec_validates_shapes_and_overlap() -> None:
    query = nn.Linear(8, 8)
    key = nn.Linear(8, 8)
    value = nn.Linear(8, 8)
    output = nn.Linear(7, 8)
    owner = nn.Module()
    owner.query = query
    owner.key = key
    owner.value = value
    owner.output = output
    spec = SeparateQKVAttentionSpec(
        query_projection=query,
        key_projection=key,
        value_projection=value,
        output_projection=output,
        attention_module=owner,
        num_heads=2,
        prune_heads=lambda module, positions: None,
    )

    with pytest.raises(ValueError, match="output projection input width"):
        canonicalize_groups((), attention_specs=(spec,))

    duplicate_owner = nn.Module()
    duplicate_owner.query = query
    duplicate_owner.key = key
    duplicate_owner.value = value
    duplicate = SeparateQKVAttentionSpec(
        query_projection=query,
        key_projection=key,
        value_projection=value,
        output_projection=query,
        attention_module=duplicate_owner,
        num_heads=2,
        prune_heads=lambda module, positions: None,
    )
    with pytest.raises(ValueError, match="must be distinct"):
        canonicalize_groups((), attention_specs=(duplicate,))

    unowned = SeparateQKVAttentionSpec(
        query_projection=query,
        key_projection=key,
        value_projection=value,
        output_projection=output,
        attention_module=nn.Module(),
        num_heads=2,
        prune_heads=lambda module, positions: None,
    )
    with pytest.raises(ValueError, match="must own"):
        canonicalize_groups((), attention_specs=(unowned,))


def test_bert_config_exposes_only_heads_and_ffn_neurons() -> None:
    model, inputs = _tiny_bert()
    tracker = _tracker(model, inputs)

    head_groups = [
        group for group in tracker.canonical_groups if group.unit_kind == UnitKind.HEAD
    ]
    ffn_groups = [
        group for group in tracker.canonical_groups if group.unit_kind != UnitKind.HEAD
    ]

    assert [group.length for group in head_groups] == [4, 4]
    assert [group.length for group in ffn_groups] == [24, 24]
    assert sum(group.length for group in head_groups) == 8
    assert sum(group.length for group in ffn_groups) == 48
    assert len(tracker.canonical_groups) == 4
    text = tracker.view_structures()
    assert "qa_outputs" not in text
    assert "bert.embeddings" not in text
    assert "attention:prune_heads:head" not in text


def test_bert_hand_calculated_l2_param_normalization_and_group_lasso() -> None:
    model, inputs = _tiny_bert()
    tracker = _tracker(model, inputs)
    _fill_target_weights(tracker, 1.0)

    l2 = tracker.get_calculation(CalcType.L2_NORM_PR_UNIT)()
    param_pr_unit = tracker.get_calculation(CalcType.PARAM_PR_UNIT)()
    expected_l2 = torch.cat(
        (
            torch.full((4,), 16.0),
            torch.full((24,), torch.sqrt(torch.tensor(32.0)).item()),
            torch.full((4,), 16.0),
            torch.full((24,), torch.sqrt(torch.tensor(32.0)).item()),
        )
    )
    expected_param = torch.cat(
        (
            torch.full((4,), 256.0),
            torch.full((24,), 32.0),
            torch.full((4,), 256.0),
            torch.full((24,), 32.0),
        )
    )

    torch.testing.assert_close(l2, expected_l2)
    torch.testing.assert_close(param_pr_unit, expected_param)
    group_lasso = tracker.create_regularizer("group_lasso")()
    torch.testing.assert_close(group_lasso, torch.tensor(3584.0))


def test_bert_partial_head_zero_stays_active_until_all_slices_are_zero() -> None:
    model, inputs = _tiny_bert()
    tracker = _tracker(model, inputs)
    group = tracker.canonical_groups[0]
    spec = group.attention_spec
    assert spec is not None

    with torch.no_grad():
        spec.query_projection.weight[:4].zero_()
        spec.query_projection.bias[:4].zero_()

    active = tracker.get_calculation(CalcType.UNIT_ACTIVE_MASK)()
    assert active[0].item() == 1.0

    tracker.fake_prune_unit(0, 0)
    active = tracker.get_calculation(CalcType.UNIT_ACTIVE_MASK)()
    assert active[0].item() == 0.0


def test_bert_physical_summary_counts_biases_without_changing_param_unit() -> None:
    model, inputs = _tiny_bert()
    tracker = _tracker(model, inputs)
    tracker.fake_prune_unit(0, 0)
    tracker.fake_prune_unit(1, 0)

    metrics = tracker.create_tracker("group_pruning_summary").track()[
        "group_pruning_summary"
    ]
    head = metrics["groups"]["bert.encoder.layer.0.attention:prune_heads:head"]
    ffn = metrics["groups"][
        "bert.encoder.layer.0.intermediate.dense:prune_out_channels"
    ]

    assert head == {
        "pruned_units": 1.0,
        "pruned_params": 256.0,
        "pruned_bias_params": 12.0,
        "pruned_physical_params": 268.0,
    }
    assert ffn == {
        "pruned_units": 1.0,
        "pruned_params": 32.0,
        "pruned_bias_params": 1.0,
        "pruned_physical_params": 33.0,
    }
    assert metrics["pruned_params"] == 288.0
    assert metrics["pruned_bias_params"] == 13.0
    assert metrics["pruned_physical_params"] == 301.0


def test_bert_group_identity_survives_fake_w4_parametrizations() -> None:
    model, inputs = _tiny_bert()
    before = _tracker(model, inputs)
    before_signature = _group_signature(before)

    for module in _target_modules(before):
        parametrize.register_parametrization(module, "weight", FakeW4())

    after = _tracker(model, inputs)
    assert _group_signature(after) == before_signature
    l2 = after.get_calculation(CalcType.L2_NORM_PR_UNIT)()
    layer = model.bert.encoder.layer[0]
    expected_head_l2 = torch.sqrt(
        layer.attention.self.query.weight[:4].square().sum()
        + layer.attention.self.key.weight[:4].square().sum()
        + layer.attention.self.value.weight[:4].square().sum()
        + layer.attention.output.dense.weight[:, :4].square().sum()
    )
    expected_ffn_l2 = torch.sqrt(
        layer.intermediate.dense.weight[0].square().sum()
        + layer.output.dense.weight[:, 0].square().sum()
    )
    torch.testing.assert_close(l2[0], expected_head_l2)
    torch.testing.assert_close(l2[4], expected_ffn_l2)
    assert torch.isfinite(l2).all()
    assert l2.requires_grad


def test_callback_backed_head_uses_public_prune_unit_surface() -> None:
    model, inputs = _tiny_bert()
    tracker = _tracker(model, inputs)

    with pytest.raises(ValueError, match="Use prune_unit"):
        tracker.get_prune_unit(0, 0)


@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is not available"
            ),
        ),
    ),
)
def test_bert_fake_and_physical_head_ffn_pruning_have_logit_parity(
    device: str,
) -> None:
    base, inputs = _tiny_bert(device)
    fake_model = copy.deepcopy(base)
    physical_model = copy.deepcopy(base)
    fake_tracker = _tracker(fake_model, inputs)
    physical_tracker = _tracker(physical_model, inputs)
    before_parameters = _num_parameters(physical_model)

    fake_tracker.fake_prune_unit(0, 1)
    fake_tracker.fake_prune_unit(1, 2)
    physical_tracker.prune_unit(0, 1)
    physical_tracker.prune_unit(1, 2)

    with torch.no_grad():
        fake_outputs = fake_model(*inputs)
        physical_outputs = physical_model(*inputs)

    torch.testing.assert_close(
        fake_outputs.start_logits,
        physical_outputs.start_logits,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        fake_outputs.end_logits,
        physical_outputs.end_logits,
        atol=1e-5,
        rtol=1e-5,
    )
    assert before_parameters - _num_parameters(physical_model) == 301
    assert [group.length for group in physical_tracker.canonical_groups] == [
        3,
        23,
        4,
        24,
    ]


def test_bert_repeated_head_pruning_maps_current_positions_to_original_ids() -> None:
    model, inputs = _tiny_bert()
    tracker = _tracker(model, inputs)

    tracker.prune_unit(0, 0)
    tracker.prune_unit(0, 0)

    attention = model.bert.encoder.layer[0].attention
    assert attention.self.num_attention_heads == 2
    assert attention.pruned_heads == {0, 1}
    assert set(model.config.pruned_heads[0]) == {0, 1}
    assert tracker.canonical_groups[0].length == 2


def test_bert_ignore_prune_protects_callback_owned_attention() -> None:
    model, inputs = _tiny_bert()
    tracker = _tracker(model, inputs)
    owner = model.bert.encoder.layer[0].attention
    tracker.fake_prune_unit(0, 0)

    result = tracker.prune_zero_structures(ignore_prune=[owner])

    assert result.pruned_units == 0
    assert owner.self.num_attention_heads == 4


def test_bert_compact_models_round_trip(tmp_path) -> None:
    model, inputs = _tiny_bert()
    head_only = copy.deepcopy(model)
    head_tracker = _tracker(head_only, inputs)
    head_tracker.prune_unit(0, 1)
    head_dir = tmp_path / "head_only"
    head_only.save_pretrained(head_dir, safe_serialization=False)
    reloaded_head = BertForQuestionAnswering.from_pretrained(head_dir).eval()

    with torch.no_grad():
        expected_head = head_only(*inputs)
        actual_head = reloaded_head(*inputs)
    torch.testing.assert_close(expected_head.start_logits, actual_head.start_logits)
    torch.testing.assert_close(expected_head.end_logits, actual_head.end_logits)

    compact = copy.deepcopy(model)
    compact_tracker = _tracker(compact, inputs)
    compact_tracker.prune_unit(0, 1)
    compact_tracker.prune_unit(1, 2)
    compact_path = tmp_path / "compact.pt"
    torch.save(compact, compact_path)
    reloaded_compact = torch.load(
        compact_path,
        map_location="cpu",
        weights_only=False,
    ).eval()

    with torch.no_grad():
        expected_compact = compact(*inputs)
        actual_compact = reloaded_compact(*inputs)
    torch.testing.assert_close(
        expected_compact.start_logits,
        actual_compact.start_logits,
    )
    torch.testing.assert_close(
        expected_compact.end_logits,
        actual_compact.end_logits,
    )
