from enum import Enum
from typing import List

from torch_structracker.torch_pruning.dependency import constants
from torch_structracker.torch_pruning.dependency.group import Group
from torch_structracker.torch_pruning.ops import OPTYPE
from torch_structracker.torch_pruning.pruner.function import prune_linear_out_channels
from torch_structracker.weight_operations import pruner_to_operation_map
from torch_structracker.weight_reducers import ParameterExtractor, WeightOperation, WeightReducer


class structureAxis(Enum):
    OUT = 0
    IN = 1
    


def initialize_from_groups(groups: List[Group]): 
    reductions = {}
    unit_count = 0
    for group in groups:
        for member in group.items:    
            if should_process_type(member.dep.target.type):
                reduction, mapping = createSpec(member,unit_count)
                add_reduction(reduction,mapping,reductions)
            else: 
                continue
        unit_count += len(group[0].root_idxs)    
    return reductions, unit_count

def should_process_type(type: OPTYPE):
    if type == OPTYPE.LINEAR:
        return True 
    else:
        return False 
                
def apply_tp_index_mapping(root_idx: list[int],tp_index_mapping):
    if not all(index_map == constants.INDEX_MAPPING_PLACEHOLDER for index_map in tp_index_mapping):
        raise ValueError("Cannot handle index mapping yet")
    return root_idx

def createSpec(member, offset) -> tuple[WeightReducer,list[int]]:
    module = member.dep.target.module
    handler = member.dep.handler
    tp_index_mapping = member.dep.index_mapping

    # Should propably have some checks here
    group_root_idxs = apply_tp_index_mapping(member.root_idxs, tp_index_mapping)
    global_idxs = [idx + offset for idx in group_root_idxs] 

    param_extractor = ParameterExtractor(module=module) 
    operation =  pruner_to_operation_map(handler=handler,task="sum")
    
    return WeightReducer(parameter_extractor=param_extractor,operation=operation),global_idxs


def add_reduction(reduction,mapping: list[int], reductions_dict):
    if reduction in reductions_dict:
        # add mapping
        existing_mapping = reductions_dict[reduction][1]
        existing_mapping.extend(mapping)
    else:
        reductions_dict[reduction] = (reduction,mapping)
