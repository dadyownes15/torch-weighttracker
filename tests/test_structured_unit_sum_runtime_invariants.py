import torch
import torch.nn as nn

from torch_structracker.calculations import StructuredUnitSum
from torch_structracker.operations import (
    QKVSourceOperation,
    WeightOperation,
    WeightOperationType,
)
from torch_structracker.reducer_plan import (
    ReducerMapping,
    ReducerPlan,
    compile_reducer_plan_from_groups,
)
from torch_structracker.reducers import ParameterExtractor, WeightReducer
from torch_structracker.torch_pruning.dependency import DependencyGraph


class CountingRowSum(WeightOperation):
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.grad_enabled_values: list[bool] = []

    def forward(self, weight):
        self.call_count += 1
        self.grad_enabled_values.append(torch.is_grad_enabled())
        return weight.sum(dim=1)


class CountingQKVSourceOperation(QKVSourceOperation):
    def __init__(self):
        super().__init__(WeightOperationType.SUM)
        self.call_count = 0

    def forward(self, weight):
        self.call_count += 1
        return super().forward(weight)


class DirectFusedMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(4, 2, batch_first=True, bias=False)

    def forward(self, x):
        output, _ = self.mha(x, x, x, need_weights=False)
        return output


def buffer_snapshot(module):
    return {
        name: (
            tuple(buffer.shape),
            buffer.dtype,
            buffer.device,
            buffer.data_ptr(),
        )
        for name, buffer in module.named_buffers()
    }


def make_counted_plan():
    first = nn.Linear(2, 2, bias=False)
    second = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        first.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        second.weight.copy_(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))

    first_operation = CountingRowSum()
    second_operation = CountingRowSum()
    first_reducer = WeightReducer(
        parameter_extractor=ParameterExtractor(first),
        operation=first_operation,
    )
    second_reducer = WeightReducer(
        parameter_extractor=ParameterExtractor(second),
        operation=second_operation,
    )
    plan = ReducerPlan(
        output_length=2,
        mappings=(
            ReducerMapping(
                reducer=first_reducer,
                destination_indices=(0, 1),
            ),
            ReducerMapping(
                reducer=second_reducer,
                destination_indices=(0, 1),
            ),
        ),
    )

    return plan, (first_operation, second_operation)


def test_structured_unit_sum_runs_each_reducer_once_per_forward():
    plan, operations = make_counted_plan()
    calculation = StructuredUnitSum(plan)

    first_result = calculation()
    second_result = calculation()
    third_result = calculation()

    assert [operation.call_count for operation in operations] == [3, 3]
    assert all(
        grad_enabled is False
        for operation in operations
        for grad_enabled in operation.grad_enabled_values
    )
    assert first_result.data_ptr() == second_result.data_ptr()
    assert second_result.data_ptr() == third_result.data_ptr()
    torch.testing.assert_close(third_result, torch.tensor([14.0, 22.0]))


def test_structured_unit_sum_reuses_destination_and_accumulator_buffers():
    plan, _ = make_counted_plan()
    calculation = StructuredUnitSum(plan)

    before = buffer_snapshot(calculation)
    destination_ptrs = tuple(dst.data_ptr() for dst in calculation.destination_indices)
    accumulator_ptr = calculation.accumulator.data_ptr()

    calculation()
    after_first = buffer_snapshot(calculation)
    calculation()
    after_second = buffer_snapshot(calculation)

    assert before == after_first == after_second
    assert destination_ptrs == tuple(
        dst.data_ptr() for dst in calculation.destination_indices
    )
    assert accumulator_ptr == calculation.accumulator.data_ptr()


def test_structured_unit_sum_runs_real_qkv_reducer_once_per_forward():
    model = DirectFusedMHA().eval()
    graph = DependencyGraph().build_dependency(
        model=model,
        example_inputs=torch.ones(2, 3, 4),
    )
    groups = list(graph.get_all_groups(root_module_types=[nn.MultiheadAttention]))
    plan = compile_reducer_plan_from_groups(
        groups,
        operation_type=WeightOperationType.SUM,
    )
    qkv_operation = CountingQKVSourceOperation()
    plan.mappings[0].reducer.operation = qkv_operation
    calculation = StructuredUnitSum(plan)

    calculation()
    calculation()

    assert len(plan.mappings) == 1
    assert qkv_operation.call_count == 2


def test_structured_unit_sum_forward_does_not_rebuild_destination_tensors(monkeypatch):
    plan, _ = make_counted_plan()
    calculation = StructuredUnitSum(plan)
    expected = torch.tensor([14.0, 22.0])

    def fail_tensor_creation(*args, **kwargs):
        raise AssertionError("forward should reuse registered destination buffers")

    monkeypatch.setattr(torch, "tensor", fail_tensor_creation)

    torch.testing.assert_close(calculation(), expected)
