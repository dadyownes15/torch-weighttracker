"""Base pruner implementation for structural pruning."""

import math
import typing
import warnings

import torch
import torch.nn as nn

from .. import dependency, ops
from .. import function

class StructurCounter:
    """Meta pruner for structural pruning.
    
    This class implements the group-level pruning strategy powered by Dependency Graph.
    See https://arxiv.org/abs/2301.12900 for details.

    Args:
        model (nn.Module): A to-be-pruned model
        example_inputs (torch.Tensor or List): dummy inputs for graph tracing.
        importance (Callable): importance estimator. 
        global_pruning (bool): enable global pruning. Default: False.
        pruning_ratio (float): global channel sparisty. Also known as pruning ratio. Default: 0.5.
        pruning_ratio_dict (Dict[nn.Module|Tuple[nn.Module], float]): layer-specific pruning ratio. Default: None. The key of the dict can be a single module or a tuple of modules. The pruning ratio will be shared by all modules in the tuple.
        max_pruning_ratio (float): the maximum pruning ratio. Default: 1.0.
        iterative_steps (int): number of steps for iterative pruning. Default: 1.
        iterative_pruning_ratio_scheduler (Callable): scheduler for iterative pruning. Default: linear_scheduler.
        ignored_layers (List[nn.Module | typing.Type]): ignored modules. Default: None.
        round_to (int): round channels to the nearest multiple of round_to. E.g., round_to=8 means channels will be rounded to 8x. Default: None.
        isomorphic (bool): enable isomorphic pruning. Default: False. https://arxiv.org/abs/2407.04616
        in_channel_groups (Dict[nn.Module, int]): The number of channel groups for layer input. Default: dict().
        out_channel_groups (Dict[nn.Module, int]): The number of channel groups for layer output. Default: dict().
        num_heads (Dict[nn.Module, int]): The number of heads for multi-head attention. Default: dict().
        prune_num_heads (bool): remove entire heads in multi-head attention. Default: False.
        prune_head_dims (bool): remove head dimensions in multi-head attention. Default: True.
        head_pruning_ratio (float): head pruning ratio. Default: 0.0.
        head_pruning_ratio_dict (Dict[nn.Module, float]): layer-specific head pruning ratio. Default: None.
        customized_pruners (dict): a dict containing module-pruner pairs. Default: None.
        unwrapped_parameters (dict): a dict containing unwrapped parameters & pruning dims. Default: None.
        root_module_types (list): types of prunable modules. Default: [nn.Conv2d, nn.Linear, nn.LSTM].
        forward_fn (Callable): A function to execute model.forward. Default: None.
        output_transform (Callable): A function to transform network outputs. Default: None.
        channel_groups (Dict[nn.Module, int]): output channel grouping. Default: dict().
        ch_sparsity (float): the same as pruning_ratio. Default: None.
        ch_sparsity_dict (Dict[nn.Module, float]): the same as pruning_ratio_dict. Default: None.
        """

    def __init__(
        self,
        # Basic
        model: nn.Module, # a simple pytorch model
        example_inputs: torch.Tensor, # a dummy input for graph tracing. Should be on the same 
        ignored_layers: typing.List[nn.Module] = None, # ignored layers
        round_to: int = None,  # round channels to the nearest multiple of round_to
        # Advanced
        in_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer input
        out_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer output
        num_heads: typing.Dict[nn.Module, int] = dict(), # The number of heads for multi-head attention
        prune_num_heads: bool = False, # remove entire heads in multi-head attention
        prune_head_dims: bool = True, # remove head dimensions in multi-head attention
        head_pruning_ratio: float = 0.0, # head pruning ratio
        head_pruning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific head pruning ratio
        customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None, # pruners for customized layers. E.g., {nn.Linear: my_linear_pruner}
        unwrapped_parameters: typing.Dict[nn.Parameter, int] = None, # unwrapped nn.Parameters & pruning_dims. For example, {ViT.pos_emb: 0}
        root_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],  # root module for each group
        forward_fn: typing.Callable = None, # a function to execute model.forward
        output_transform: typing.Callable = None, # a function to transform network outputs
        
    ):
        self.model = model

        if len(num_heads) > 0:
            out_channel_groups.update(num_heads)

        self.in_channel_groups = in_channel_groups
        self.out_channel_groups = out_channel_groups
        self.root_module_types = root_module_types
        self.round_to = round_to

        # MHA
        self.num_heads = num_heads
        self.prune_num_heads = prune_num_heads
        self.prune_head_dims = prune_head_dims
        self.head_pruning_ratio = head_pruning_ratio

        ###############################################
        # Ignored layers and submodules
        self.ignored_layers = []
        self.ignored_params = []
        if ignored_layers is not None:
            for layer in ignored_layers:
                if isinstance(layer, nn.Module):
                    self.ignored_layers.extend(list(layer.modules()))
                elif isinstance(layer, nn.Parameter):
                    self.ignored_params.append(layer)

        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs=example_inputs,
            forward_fn=forward_fn,
            output_transform=output_transform,
            unwrapped_parameters=unwrapped_parameters,
            customized_pruners=customized_pruners,
            ignored_params=self.ignored_params,
        )

        ###############################################
        # Detect group convs & group norms
        for m in self.model.modules():
            layer_pruner = self.DG.get_pruner_of_module(m)
            in_ch_group = layer_pruner.get_in_channel_groups(m)
            out_ch_group = layer_pruner.get_out_channel_groups(m)
            if isinstance(m, ops.TORCH_CONV) and m.groups == m.out_channels:
                continue
            if in_ch_group > 1:
                self.in_channel_groups[m] = in_ch_group
            if out_ch_group > 1:
                self.out_channel_groups[m] = out_ch_group

        ###############################################
        # Initial channels/dims of each layer
        self.layer_init_out_ch = {}
        self.layer_init_in_ch = {}
        self.init_num_heads = {}
        for m in self.DG.module2node.keys():
            if ops.module2type(m) in self.DG.REGISTERED_PRUNERS:
                self.layer_init_out_ch[m] = self.DG.get_out_channels(m)
                self.layer_init_in_ch[m] = self.DG.get_in_channels(m)
                if m in self.num_heads:
                    self.init_num_heads[m] = self.num_heads[m]

        ###############################################
        # Count the number of total channels at initialization
        # if self.global_pruning:
        initial_total_channels = 0
        initial_total_heads = 0
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types):
            _is_atten, qkv_layers = self._is_atten_group(group)
            if _is_atten:
                group = self._downstream_node_as_root_if_attention(group)
                if group is None:
                    continue
            initial_total_channels += ((self.DG.get_out_channels(
                group[0][0].target.module)) // self._get_channel_groups(group))
            for dep, _ in group:
                if dep.target.module in self.num_heads and self.DG.is_out_channel_pruning_fn(dep.handler):
                    initial_total_heads += self.num_heads[dep.target.module]
                    break  # only count heads once
        self.initial_total_channels = initial_total_channels
        self.initial_total_heads = initial_total_heads

    def step(self, interactive: bool = False) -> typing.Union[typing.Generator, None]:
        """Execute one step of pruning.
        
        Args:
            interactive: If True, yields groups for interactive pruning. 
                        If False, prunes all groups automatically.
                        
        Returns:
            Generator yielding pruning groups if interactive=True, None otherwise.
        """
        self.current_step += 1
        if interactive:  # yield groups for interactive pruning
            return self._prune()
        else:
            for group in self._prune():
                group.prune()

    def reset(self) -> None:
        self.current_step = 0

    def _is_atten_group(self, group) -> bool:
        is_attn = False
        qkv_layers = []
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            if self.DG.is_out_channel_pruning_fn(pruning_fn) and module in self.num_heads:
                qkv_layers.append(module)
                is_attn = True
        return is_attn, qkv_layers

    def _get_channel_groups(self, group) -> int:
        ch_groups = []
        # has_unbind = False
        # unbind_node = None

        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            channel_groups = self.out_channel_groups if self.DG.is_out_channel_pruning_fn(
                pruning_fn) else self.in_channel_groups

            if module in channel_groups:
                ch_groups.append(channel_groups[module])

            # if dep.source.type==ops.OPTYPE.UNBIND:
            #    has_unbind = True
            #    unbind_node = dep.source

        # if has_unbind and ch_groups>1:
        #    ch_groups = ch_groups // len(unbind_node.outputs)
        if len(ch_groups) == 0:
            return 1
        return max(ch_groups)  # no channel grouping

    def _downstream_node_as_root_if_attention(self, group):
        # Use a downstream node as the root if torch.unbind exists. TODO: find a general way to handle torch.unbind in timm
        is_attention = False
        downstream_dep = None
        for _dep, _idxs in group:
            if _dep.source.module in self.num_heads and self.DG.is_out_channel_pruning_fn(_dep.handler):
                is_attention = True
            if isinstance(_dep.target.module, tuple(self.root_module_types)) and self.DG.is_in_channel_pruning_fn(_dep.handler):
                downstream_dep = _dep
                idxs = _idxs
        # use a downstream node as the root node for attention layers
        if is_attention and downstream_dep is not None:
            group = self.DG.get_pruning_group(
                downstream_dep.target.module, downstream_dep.handler, idxs)
            return group
        return None

    def _round_to(self, n_pruned, current_channels, round_to):
        rounded_channels = current_channels - n_pruned
        rounded_channels = rounded_channels - rounded_channels % round_to
        n_pruned = current_channels - rounded_channels
        return max(n_pruned, 0)

    @torch.no_grad
    def prune_group(self, group):
        pass