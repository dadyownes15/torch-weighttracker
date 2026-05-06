from typing import List

from torch_structracker.torch_pruning.dependency import constants
from torch_structracker.torch_pruning.dependency.group import Group
from torch_structracker.torch_pruning.ops import OPTYPE
from torch_structracker.torch_pruning.pruner.function import prune_linear_out_channels
from torch_structracker.weight_operations import pruner_to_operation_map
from torch_structracker.weight_reducers import ParameterExtractor, WeightOperation, WeightReducer


def initialize_from_groups(groups: List[Group]): 
    ops = {}
    for group_idx,group in enumerate(groups):
        for member in group.items:    
            if should_process_type(member.dep.target.type):
                op, mapping = createSpec(member,group_idx)
                add_operation(op,mapping,ops)
            else: 
                continue
            
    return ops

def should_process_type(type: OPTYPE):
    if type == OPTYPE.LINEAR:
        return True 
    else:
        return False 
                
def apply_tp_index_mapping(root_idx: list[int],tp_index_mapping):
    if not all(index_map == constants.INDEX_MAPPING_PLACEHOLDER for index_map in tp_index_mapping):
        raise ValueError("Cannot handle index mapping yet")
    return root_idx

def createSpec(member, group_idx) -> tuple[WeightReducer,list[int]]:
    module = member.dep.target.module
    handler = member.dep.handler
    tp_index_mapping = member.dep.index_mapping

    # Should propably have some checks here
    group_root_idxs = apply_tp_index_mapping(member.root_idxs, tp_index_mapping)
    global_idxs = [idx + group_idx for idx in group_root_idxs] 

    param_extractor = ParameterExtractor(module=module) 
    operation =  pruner_to_operation_map(handler=handler,task="sum")
    
    return WeightReducer(parameter_extractor=param_extractor,operation=operation),global_idxs


def add_operation(op,mapping: list[int], ops_dict):
    if op in ops_dict:
        # add mapping
        existing_mapping = ops_dict[op][1]
        existing_mapping.extend(mapping)
    else:
        ops_dict[op] = (op,mapping)
"""

<bound method LinearPruner.prune_out_channels of <torch_structracker.torch_pruning.pruner.function.LinearPruner object at 0x12ecf7980>>
{'trigger': <bound method LinearPruner.prune_out_channels of <torch_structracker.torch_pruning.pruner.function.LinearPruner object at 0x12ecf7980>>, 'handler': <bound method LinearPruner.prune_out_channels of <torch_structracker.torch_pruning.pruner.function.LinearPruner object at 0x12ecf7980>>, 'source': <Node: (first_layer (Linear(in_features=1, out_features=3, bias=True)))>, 'target': <Node: (first_layer (Linear(in_features=1, out_features=3, bias=True)))>, 'index_mapping': [None, None]}
dep 2
{'trigger': <bound method LinearPruner.prune_out_channels of <torch_structracker.torch_pruning.pruner.function.LinearPruner object at 0x12ecf7980>>, 'handler': <bound method DummyPruner.prune_out_channels of <torch_structracker.torch_pruning.ops.ReshapePruner object at 0x12ecf7680>>, 'source': <Node: (first_layer (Linear(in_features=1, out_features=3, bias=True)))>, 'target': <Node: (_Reshape_4())>, 'index_mapping': [None, None]}
.

"""
