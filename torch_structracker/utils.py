from torch_structracker.torch_pruning.dependency.group import Group


def total_units(groups):
    units = 0
    for idx_g,group in enumerate(groups):
        units += len(group[0].idxs)

    return units


