import torch.nn as nn

from torch_structracker.operations import (
    MeanWeight,
    QKVSourceOperation,
    SumWeight,
    WeightOperationType,
)
from torch_structracker.reducers import (
    ParameterExtractor,
    ParameterTupleExtractor,
    WeightReducer,
)


def test_weight_reducer_can_be_found_in_dict_by_equivalent_reducer():
    linear = nn.Linear(2, 3)
    reducer = WeightReducer(
        parameter_extractor=ParameterExtractor(linear),
        operation=SumWeight(dim=0),
    )
    equivalent_reducer = WeightReducer(
        parameter_extractor=ParameterExtractor(linear),
        operation=SumWeight(dim=0),
    )
    reducers = {reducer: [0, 1]}

    assert equivalent_reducer in reducers
    assert reducers[equivalent_reducer] == [0, 1]


def test_weight_reducer_identity_distinguishes_operation_config():
    linear = nn.Linear(2, 3)
    reducer = WeightReducer(
        parameter_extractor=ParameterExtractor(linear),
        operation=SumWeight(dim=0),
    )
    different_dim_reducer = WeightReducer(
        parameter_extractor=ParameterExtractor(linear),
        operation=SumWeight(dim=1),
    )
    different_operation_reducer = WeightReducer(
        parameter_extractor=ParameterExtractor(linear),
        operation=MeanWeight(dim=0),
    )
    reducers = {reducer: [0, 1]}

    assert different_dim_reducer not in reducers
    assert different_operation_reducer not in reducers


def test_weight_reducer_identity_distinguishes_modules():
    reducer = WeightReducer(
        parameter_extractor=ParameterExtractor(nn.Linear(2, 3)),
        operation=SumWeight(dim=0),
    )
    other_module_reducer = WeightReducer(
        parameter_extractor=ParameterExtractor(nn.Linear(2, 3)),
        operation=SumWeight(dim=0),
    )
    reducers = {reducer: [0, 1]}

    assert other_module_reducer not in reducers


def test_weight_reducer_identity_supports_parameter_tuple_extractors():
    q_projection = nn.Linear(2, 2)
    k_projection = nn.Linear(2, 2)
    v_projection = nn.Linear(2, 2)
    reducer = WeightReducer(
        parameter_extractor=ParameterTupleExtractor(
            ParameterExtractor(q_projection),
            ParameterExtractor(k_projection),
            ParameterExtractor(v_projection),
        ),
        operation=QKVSourceOperation(
            operation_type=WeightOperationType.SUM,
        ),
    )
    equivalent_reducer = WeightReducer(
        parameter_extractor=ParameterTupleExtractor(
            ParameterExtractor(q_projection),
            ParameterExtractor(k_projection),
            ParameterExtractor(v_projection),
        ),
        operation=QKVSourceOperation(
            operation_type=WeightOperationType.SUM,
        ),
    )
    reducers = {reducer: [0]}

    assert equivalent_reducer in reducers
