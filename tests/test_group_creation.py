from types import SimpleNamespace

import torch
import torch.nn as nn

from torch_structracker.operations import SumWeight, WeightOperationType
from torch_structracker.reducer_plan import (
    ReducerMapping,
    add_mapping,
    compile_reducer_plan_from_groups,
    create_reducer_mappings_for_member,
    validate_reducer_plan,
)
from torch_structracker.reducers import ParameterExtractor, WeightReducer
from torch_structracker.torch_pruning.dependency import DependencyGraph, constants
from torch_structracker.torch_pruning.pruner.function import (
    prune_linear_out_channels,
)


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
):
    return SimpleNamespace(
        dep=SimpleNamespace(
            target=SimpleNamespace(module=module),
            handler=handler,
            index_mapping=[
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


def test_create_reducer_mappings_reads_linear_member():
    linear = nn.Linear(3, 2)
    member = make_member(
        module=linear,
        handler=prune_linear_out_channels,
        root_idxs=[0, 1],
    )

    (mapping,) = create_reducer_mappings_for_member(
        member=member,
        group_offset=3,
        operation_type=WeightOperationType.SUM,
    )

    assert isinstance(mapping.reducer, WeightReducer)
    assert mapping.reducer.parameter_extractor.module is linear
    assert mapping.reducer.parameter_extractor.name == "weight"
    assert isinstance(mapping.reducer.operation, SumWeight)
    assert mapping.reducer.operation.dim == 1
    assert mapping.destination_indices == (3, 4)


def test_add_mapping_extends_existing_reducer_mapping():
    reducer = WeightReducer(
        parameter_extractor=ParameterExtractor(nn.Linear(2, 3)),
        operation=SumWeight(dim=0),
    )
    mappings = {
        reducer: ReducerMapping(
            reducer=reducer,
            destination_indices=(0, 1),
        )
    }

    add_mapping(
        ReducerMapping(
            reducer=reducer,
            destination_indices=(2,),
        ),
        mappings,
    )

    assert mappings[reducer].destination_indices == (0, 1, 2)


def test_compile_reducer_plan_from_groups_creates_valid_linear_plan():
    _, groups = linear_groups_from_simple_mlp()

    plan = compile_reducer_plan_from_groups(
        groups,
        operation_type=WeightOperationType.SUM,
    )

    assert plan.output_length > 0
    assert len(plan.mappings) > 0
    validate_reducer_plan(plan)
