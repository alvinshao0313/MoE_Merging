import torch
import torch.nn as nn
import transformers
from collections import defaultdict, OrderedDict
import re
from models.new_moe import LoRAExpert
from models.deepseek.v1.modeling_deepseekv1 import DeepseekMoE, DeepseekMLP


def get_experts(layer, exclude_expert_names_regex: list):
    """
    get the experts in a layer
    :param layer: a layer of the model
    :return: list, experts in the layer
    """
    experts = []
    for name, module in layer.named_modules():
        exclude = any([re.search(exclude_pattern, name) for exclude_pattern in exclude_expert_names_regex])
        if not exclude and isinstance(module, DeepseekMLP):
            experts.append(module)
    return experts


def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.search(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


def average_merging(experts_to_merge: list, exclude_param_names_regex: list):
    """
    average merging method
    :param experts_to_merge: list, individual experts that need to be merged
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    # dictionary of list, where key is the parameter name,
    # value is a list of the corresponding parameters of all the models that need to be merged
    experts_to_merge_param_dict = defaultdict(list)
    # iterate each individual model that needs to be merged
    for model_to_merge in experts_to_merge:
        param_dict = {param_name: param_value for param_name, param_value in model_to_merge.named_parameters()}
        # exclude parameter whose name matches element in exclude_param_names_regex
        param_names_to_merge = get_param_names_to_merge(input_param_names=list(param_dict.keys()),
                                                        exclude_param_names_regex=exclude_param_names_regex)
        for param_name in param_names_to_merge:
            experts_to_merge_param_dict[param_name].append(param_dict[param_name])

    with torch.no_grad():
        # average merging of individual models' parameters
        averaged_params = {param_name: torch.stack(model_to_merge_param, dim=0).mean(dim=0) for
                           param_name, model_to_merge_param in experts_to_merge_param_dict.items()}

    return averaged_params


def copy_params_to_model(params: dict, model: nn.Module):
    """
    copy parameters in "params" to the model
    :param params: dict, dictionary of parameters
    :param model: nn.Module, model that needs to copy parameters
    :return:
    """
    for param_name, param_value in model.named_parameters():
        if param_name in params:
            param_value.data.copy_(params[param_name])


def update_org_experts(experts_to_merge: nn.ModuleList, avg_expert_params: dict):
    """
    get the expert vector from the experts to merge
    :param experts_to_merge: nn.ModuleList, list of experts to be merged
    :param avg_expert: dict, average expert parameters
    :return: list, expert vector
    """
    new_experts = []
    for expert in experts_to_merge:
        for param_name, param_value in expert.named_parameters():
            if param_name in avg_expert_params.keys():
                diff = param_value.data - avg_expert_params[param_name]
                param_value.data.copy_(diff)
        new_experts.append(expert)
    experts_to_merge = nn.ModuleList(new_experts)
    return experts_to_merge


def init_lora_experts(experts: nn.ModuleList, rank: int = 8):
    """
    Initialize LoRA experts with the average expert parameters.
    :param experts: nn.ModuleList, list of experts to be initialized with LoRA
    :param rank: int, rank for LoRA
    """
    new_experts = []
    for expert in experts:
        expert = LoRAExpert(
            expert=expert,
            hidden_size=expert.hidden_size,
            intermediate_size=expert.intermediate_size,
            rank=rank
        )
        new_experts.append(expert)
    experts = nn.ModuleList(new_experts)
    return experts


def get_lora_params(experts: nn.ModuleList):
    """
    Get the LoRA parameters from the experts.
    :param experts: nn.ModuleList, list of experts
    :return: list, LoRA parameters
    """
    lora_params = []
    for idx, expert in enumerate(experts):
        if isinstance(expert, LoRAExpert):
            for param_name, param_value in experts[idx].named_parameters():
                lora_params.append(param_value)
    return lora_params


def set_lora_params_requires_grad(layer: nn.Module, requires_grad: bool):
    """
    Set the requires_grad attribute of LoRA parameters.
    :param layer: nn.Module, layer containing LoRA parameters
    :param requires_grad: bool, whether to set requires_grad to True or False
    """
    for name, param in layer.named_parameters():
        if 'lora' in name:
            param.requires_grad = requires_grad
        else:
            param.requires_grad = False
