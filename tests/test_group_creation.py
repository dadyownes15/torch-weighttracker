from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from torch_structracker.torch_pruning.dependency import DependencyGraph, constants
from torch_structracker.torch_pruning.pruner.function import (
    prune_linear_in_channels,
    prune_linear_out_channels,
)
from torch_structracker.utils import (
    add_operation,
    apply_tp_index_mapping,
    createSpec,
    initialize_from_groups,
)
from torch_structracker.weight_operations import SumWeight, pruner_to_operation_map
from torch_structracker.weight_reducers import ParameterExtractor, WeightReducer


class SimpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first_layer = nn.Linear(1, 3)
        self.second_layer = nn.Linear(3, 3)
        self.third_layer = nn.Linear(3, 3)
        self.output_layer = nn.Linear(3, 1)
        self.net = nn.Sequential(
            self.first_layer,
            self.second_layer,
            self.third_layer,
            self.output_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_member(
    module: nn.Module,
    handler=prune_linear_out_channels,
    root_idxs: list[int] | None = None,
    index_mapping: list[object] | None = None,
):
    return SimpleNamespace(
        dep=SimpleNamespace(
            target=SimpleNamespace(module=module),
            handler=handler,
            index_mapping=index_mapping
            if index_mapping is not None
            else [
                constants.INDEX_MAPPING_PLACEHOLDER,
                constants.INDEX_MAPPING_PLACEHOLDER,
            ],
        ),
        root_idxs=root_idxs if root_idxs is not None else [0, 1],
    )


def linear_groups_from_simple_mlp():
    model = SimpleMLP()
    graph = DependencyGraph().build_dependency(
        model=model,
        example_inputs=torch.ones(1, 1),
    )
    return model, list(graph.get_all_groups(root_module_types=[nn.Linear]))


def linear_members_with_group_idx(groups):
    return [
        (group_idx, member)
        for group_idx, group in enumerate(groups)
        for member in group.items
        if isinstance(member.dep.target.module, nn.Linear)
    ]


def expected_operation_dim(handler):
    if handler == prune_linear_out_channels:
        return 0
    if handler == prune_linear_in_channels:
        return 1
    raise AssertionError(f"Unexpected linear pruning handler: {handler}")


def test_apply_tp_index_mapping_returns_root_indices_for_placeholders():
    root_idxs = [0, 2]

    result = apply_tp_index_mapping(
        root_idxs,
        [
            constants.INDEX_MAPPING_PLACEHOLDER,
            constants.INDEX_MAPPING_PLACEHOLDER,
        ],
    )

    assert result == root_idxs


def test_apply_tp_index_mapping_raises_for_real_mapping():
    with pytest.raises(ValueError, match="Cannot handle index mapping yet"):
        apply_tp_index_mapping([0, 2], [constants.INDEX_MAPPING_PLACEHOLDER, 1])


def test_create_spec_reads_module_handler_and_index_mapping_from_linear_member():
    linear = nn.Linear(3, 2)
    member = make_member(
        module=linear,
        handler=prune_linear_out_channels,
        root_idxs=[0, 1],
    )

    reducer, mapping = createSpec(member, group_idx=3)

    assert isinstance(reducer, WeightReducer)
    assert reducer.parameter_extractor.module is linear
    assert reducer.parameter_extractor.name == "weight"
    assert isinstance(reducer.operation, SumWeight)
    assert reducer.operation.dim == 0
    assert mapping == [3, 4]


def test_pruner_to_operation_map_linear_out_channels():
    operation = pruner_to_operation_map(prune_linear_out_channels, task="sum")

    assert isinstance(operation, SumWeight)
    assert operation.dim == 0


def test_pruner_to_operation_map_linear_in_channels():
    operation = pruner_to_operation_map(prune_linear_in_channels, task="sum")

    assert isinstance(operation, SumWeight)
    assert operation.dim == 1


def test_parameter_extractor_returns_linear_weight():
    linear = nn.Linear(2, 3, bias=False)

    extractor = ParameterExtractor(module=linear)

    assert extractor.get() is linear.weight


def test_add_operation_adds_new_operation():
    op = WeightReducer(
        parameter_extractor=ParameterExtractor(nn.Linear(2, 3)),
        operation=SumWeight(dim=0),
    )
    ops = {}

    add_operation(op, [0, 1], ops)

    assert ops[op] == (op, [0, 1])


def test_add_operation_extends_existing_operation_mapping():
    op = WeightReducer(
        parameter_extractor=ParameterExtractor(nn.Linear(2, 3)),
        operation=SumWeight(dim=0),
    )
    ops = {op: (op, [0, 1])}

    add_operation(op, [2], ops)

    assert ops[op] == (op, [0, 1, 2])


def test_initialize_from_groups_creates_linear_reducers_from_simple_mlp_groups():
    _, groups = linear_groups_from_simple_mlp()
    expected_members = linear_members_with_group_idx(groups)

    ops = initialize_from_groups(groups)

    assert len(ops) == len(expected_members)
    for (reducer, mapping), (group_idx, member) in zip(
        ops.values(),
        expected_members,
        strict=True,
    ):
        assert isinstance(reducer, WeightReducer)
        assert isinstance(reducer.parameter_extractor.module, nn.Linear)
        assert reducer.parameter_extractor.module is member.dep.target.module
        assert isinstance(reducer.operation, SumWeight)
        assert reducer.operation.dim == expected_operation_dim(member.dep.handler)
        assert mapping == [idx + group_idx for idx in member.root_idxs]
