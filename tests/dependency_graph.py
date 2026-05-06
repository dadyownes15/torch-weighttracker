from __future__ import annotations


import pytest
import torch.nn as nn
import torch 
from torch_structracker.torch_pruning.dependency import DependencyGraph
from torch_structracker.torch_pruning.pruner.function import LinearPruner, prune_linear_out_channels 

class simpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first_layer =nn.Linear(1,3)
        self.secound_layer =nn.Linear(3,3)
        self.third_layer =nn.Linear(3,1) 
        self.net = nn.Sequential(
            self.first_layer,
            self.secound_layer,
            self.third_layer
        )

    def forward(self,x):
        return self.net(x)


def test_specs_creation():
    model = simpleMLP()
    input_ex = torch.tensor([[1.0]])
    print(model.forward(input_ex))
    graph = DependencyGraph().build_dependency(model = model,  example_inputs=input_ex)
    all_groups = []

    for group in graph.get_all_groups():
        all_groups.append(group)
    # iteraer
    first_group = all_groups[2] 
    first_item = first_group[0]
    print("first item: ",first_item)
    dep = first_item[0]
    dep2 = first_group[1][0]
    print(dep.handler)
    print(vars(dep))
    print("dep 2")
    print(vars(dep2))
    
    assert dep.handler == prune_linear_out_channels
