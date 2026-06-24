import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_weighttracker import WeightTracker
from torch_weighttracker.canonical_units import UnitAxis
from torch_weighttracker.torch_pruning.dependency import DependencyGraph
from torch_weighttracker.torch_pruning.pruner.function import BasePruningFunc


class TwoLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=False)
        self.fc2 = nn.Linear(3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class LinearNoopPruner(BasePruningFunc):
    TARGET_MODULES = nn.Linear

    def prune_out_channels(self, layer: nn.Linear, idxs) -> nn.Linear:
        return layer

    def prune_in_channels(self, layer: nn.Linear, idxs) -> nn.Linear:
        return layer

    def get_out_channels(self, layer: nn.Linear) -> int:
        return layer.out_features

    def get_in_channels(self, layer: nn.Linear) -> int:
        return layer.in_features


class AmbiguousLinearPruner(BasePruningFunc):
    TARGET_MODULES = nn.Linear

    def prune_out_channels(self, layer: nn.Linear, idxs) -> nn.Linear:
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer: nn.Linear) -> int:
        return layer.out_features

    def get_in_channels(self, layer: nn.Linear) -> int:
        return layer.in_features


def _canonical_members_for_module(tracker: WeightTracker, module: nn.Module):
    return tuple(
        member
        for group in tracker.canonical_groups
        for member in group.members
        if member.module is module
    )


def test_dependency_graph_prefers_instance_customized_pruner() -> None:
    model = TwoLinear()
    class_pruner = LinearNoopPruner()
    instance_pruner = LinearNoopPruner()

    graph = DependencyGraph().build_dependency(
        model=model,
        example_inputs=torch.ones(1, 2),
        customized_pruners={
            nn.Linear: class_pruner,
            model.fc1: instance_pruner,
        },
    )

    assert graph.get_pruner_of_module(model.fc1) is instance_pruner
    assert graph.get_pruner_of_module(model.fc2) is class_pruner
    assert list(graph.get_all_groups(root_module_types=[nn.Linear]))


def test_dependency_graph_class_customized_pruner_still_works() -> None:
    model = TwoLinear()
    class_pruner = LinearNoopPruner()

    graph = DependencyGraph().build_dependency(
        model=model,
        example_inputs=torch.ones(1, 2),
        customized_pruners={nn.Linear: class_pruner},
    )

    assert graph.get_pruner_of_module(model.fc1) is class_pruner
    assert graph.get_pruner_of_module(model.fc2) is class_pruner
    assert list(graph.get_all_groups(root_module_types=[nn.Linear]))


def test_weight_tracker_canonicalizes_instance_custom_linear_pruner() -> None:
    model = TwoLinear()
    class_pruner = LinearNoopPruner()
    instance_pruner = LinearNoopPruner()

    tracker = WeightTracker(
        model,
        example_inputs=torch.ones(1, 2),
        customized_pruners={
            nn.Linear: class_pruner,
            model.fc1: instance_pruner,
        },
    )

    assert tracker.dependency_graph.get_pruner_of_module(model.fc1) is instance_pruner
    assert tracker.dependency_graph.get_pruner_of_module(model.fc2) is class_pruner

    fc1_members = _canonical_members_for_module(tracker, model.fc1)
    assert any(
        member.unit_axis == UnitAxis.OUT_CHANNEL
        and getattr(member.handler, "__self__", None) is instance_pruner
        for member in fc1_members
    )


def test_weight_tracker_does_not_infer_feature_for_ambiguous_custom_pruner() -> None:
    model = TwoLinear()
    ambiguous_pruner = AmbiguousLinearPruner()

    tracker = WeightTracker(
        model,
        example_inputs=torch.ones(1, 2),
        customized_pruners={model.fc1: ambiguous_pruner},
    )

    fc1_members = _canonical_members_for_module(tracker, model.fc1)
    assert UnitAxis.FEATURE not in {member.unit_axis for member in fc1_members}
    assert all(
        getattr(member.handler, "__self__", None) is not ambiguous_pruner
        for member in fc1_members
    )


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class HardcodedShortcut(nn.Module):
    def __init__(
        self,
        *,
        original_in: int,
        original_out: int,
        stride: int,
        input_keep: tuple[int, ...],
        output_keep: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.original_in = original_in
        self.original_out = original_out
        self.stride = stride
        self.input_keep = input_keep
        self.output_keep = output_keep
        self.out_channels = len(output_keep)

        input_pos = {
            original_idx: pos for pos, original_idx in enumerate(input_keep)
        }
        pairs = [
            (input_pos[original_out_idx], out_pos)
            for out_pos, original_out_idx in enumerate(output_keep)
            if original_out_idx in input_pos
        ]

        self.register_buffer(
            "src",
            torch.tensor([pair[0] for pair in pairs], dtype=torch.long),
        )
        self.register_buffer(
            "dst",
            torch.tensor([pair[1] for pair in pairs], dtype=torch.long),
        )

    def forward(self, x):
        if self.stride != 1:
            x = x[:, :, :: self.stride, :: self.stride]

        out = x.new_zeros(x.shape[0], self.out_channels, x.shape[2], x.shape[3])
        out[:, self.dst, :, :] = x[:, self.src, :, :]
        return out


class ShortcutPruner(BasePruningFunc):
    TARGET_MODULES = LambdaLayer

    def __init__(
        self,
        original_in: int,
        original_out: int,
        stride: int,
        *,
        input_keep: tuple[int, ...] | None = None,
        output_keep: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.original_in = original_in
        self.original_out = original_out
        self.stride = stride
        self.input_keep = input_keep or tuple(range(original_in))
        self.output_keep = output_keep or tuple(range(original_out))
        self.prune_in_calls: list[tuple[int, ...]] = []
        self.prune_out_calls: list[tuple[int, ...]] = []

    def prune_out_channels(self, layer: LambdaLayer, idxs) -> LambdaLayer:
        idxs = tuple(sorted(int(index) for index in idxs))
        self.prune_out_calls.append(idxs)
        self.output_keep = _remove_positions(self.output_keep, idxs)
        self._install(layer)
        return layer

    def prune_in_channels(self, layer: LambdaLayer, idxs) -> LambdaLayer:
        idxs = tuple(sorted(int(index) for index in idxs))
        self.prune_in_calls.append(idxs)
        self.input_keep = _remove_positions(self.input_keep, idxs)
        self._install(layer)
        return layer

    def get_out_channels(self, layer: LambdaLayer) -> int:
        return len(self.output_keep)

    def get_in_channels(self, layer: LambdaLayer) -> int:
        return len(self.input_keep)

    def _install(self, layer: LambdaLayer) -> None:
        layer.lambd = HardcodedShortcut(
            original_in=self.original_in,
            original_out=self.original_out,
            stride=self.stride,
            input_keep=self.input_keep,
            output_keep=self.output_keep,
        )


def _remove_positions(values: tuple[int, ...], idxs: tuple[int, ...]):
    remove = set(idxs)
    return tuple(value for pos, value in enumerate(values) if pos not in remove)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, planes // 4, planes // 4),
                    "constant",
                    0,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.linear = nn.Linear(64, 10)

    def _make_layer(self, planes, blocks, stride):
        layers = []
        for block_stride in [stride] + [1] * (blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def resnet20():
    return ResNet()


def test_resnet20_instance_customized_pruner_prunes_downsample_shortcut():
    model = resnet20()
    x = torch.randn(1, 3, 32, 32)

    layer2_shortcut = model.layer2[0].shortcut
    layer3_shortcut = model.layer3[0].shortcut

    layer2_pruner = ShortcutPruner(16, 32, stride=2)
    layer3_pruner = ShortcutPruner(32, 64, stride=2)

    tracker = WeightTracker(
        model,
        example_inputs=x,
        customized_pruners={
            layer2_shortcut: layer2_pruner,
            layer3_shortcut: layer3_pruner,
        },
    )

    dg = tracker.dependency_graph

    assert dg.get_pruner_of_module(layer2_shortcut) is layer2_pruner
    assert dg.get_pruner_of_module(layer3_shortcut) is layer3_pruner

    group = dg.get_pruning_group(
        layer2_shortcut,
        layer2_pruner.prune_out_channels,
        [8],
    )
    group.prune()

    assert layer2_pruner.prune_out_calls == [(8,)]
    assert layer3_pruner.prune_out_calls == []
    assert isinstance(layer2_shortcut.lambd, HardcodedShortcut)

    y = model(torch.randn(1, 3, 32, 32))
    assert y.shape == (1, 10)
