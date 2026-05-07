import pytest
import timm
import torch
import torch.nn as nn

from torch_structracker.calculations import CalculationType
from torch_structracker.structure_tracker import StructureTracker


class TwoLayerTrackerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=False)
        self.fc2 = nn.Linear(3, 1, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class ContainerIgnoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(2, 3, bias=False)
        self.ignored_block = nn.Sequential(
            nn.Linear(3, 3, bias=False),
            nn.ReLU(),
            nn.Linear(3, 3, bias=False),
        )
        self.head = nn.Linear(3, 1, bias=False)

    def forward(self, x):
        return self.head(self.ignored_block(self.stem(x)))


class UnwrappedParameterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 3)
        self.scale = nn.Parameter(torch.ones(3))

    def forward(self, x):
        return self.fc(x) * self.scale


def reducer_modules(calculation):
    return {
        reducer.parameter_extractor.module
        for reducer in calculation.reducers
        if hasattr(reducer.parameter_extractor, "module")
    }


def test_ignored_layer_does_not_contribute_to_structured_unit_sum():
    model = TwoLayerTrackerModel()
    with torch.no_grad():
        model.fc1.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ]
            )
        )
        model.fc2.weight.copy_(torch.tensor([[100.0, 200.0, 300.0]]))

    tracker = StructureTracker(
        model,
        example_inputs=torch.ones(1, 2),
        root_module_types=[nn.Linear],
        ignored_layers=[model.fc2],
    )
    calculation = tracker.get_calculation(CalculationType.STRUCTURED_UNIT_SUM)

    assert calculation.output_length == 3
    assert reducer_modules(calculation) == {model.fc1}
    torch.testing.assert_close(
        calculation(),
        torch.tensor([3.0, 7.0, 11.0]),
    )


def test_ignored_container_expands_to_child_modules_before_group_creation():
    model = ContainerIgnoreModel()
    ignored_modules = set(model.ignored_block.modules())

    tracker = StructureTracker(
        model,
        example_inputs=torch.ones(1, 2),
        root_module_types=[nn.Linear],
        ignored_layers=[model.ignored_block],
    )
    calculation = tracker.get_calculation(CalculationType.STRUCTURED_UNIT_SUM)

    assert model.ignored_block[0] in tracker.ignored_layers
    assert model.ignored_block[2] in tracker.ignored_layers
    assert all(
        member.dep.target.module not in ignored_modules
        for group in tracker.groups
        for member in group.items
    )
    assert reducer_modules(calculation).isdisjoint(ignored_modules)


def test_ignored_unwrapped_parameter_is_removed_from_dependency_groups():
    model = UnwrappedParameterModel()

    tracker = StructureTracker(
        model,
        example_inputs=torch.ones(1, 2),
        root_module_types=[nn.Linear],
        unwrapped_parameters={model.scale: 0},
        ignored_params=[model.scale],
    )

    assert all(
        member.dep.target.module is not model.scale
        for group in tracker.groups
        for member in group.items
    )


def test_ignored_attention_qkv_layer_is_not_tracked_in_attention_view():
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        img_size=32,
        num_classes=0,
    ).eval()
    attention = model.blocks[0].attn

    tracker = StructureTracker(
        model,
        example_inputs=torch.ones(1, 3, 32, 32),
        root_module_types=[nn.Linear],
        ignored_layers=[attention.qkv],
        num_heads={attention.qkv: attention.num_heads},
        prune_num_heads=True,
    )

    assert all(
        member.dep.target.module is not attention.qkv
        for group in tracker.groups
        for member in group.items
    )
    with pytest.raises(ValueError, match="requires dependency groups"):
        tracker.get_calculation(CalculationType.STRUCTURED_UNIT_SUM)
