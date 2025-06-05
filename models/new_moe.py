import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
import math


class LoRAExpert(nn.Module):
    def __init__(self, expert: nn.Module, hidden_size=None, intermediate_size=None, rank=None):
        super().__init__()
        self.config = expert.config
        self.hidden_size = expert.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = expert.intermediate_size if intermediate_size is None else intermediate_size

        self.rank = rank
        if self.rank is None:
            self.gate_proj = expert.gate_proj
            self.up_proj = expert.up_proj
            self.down_proj = expert.down_proj
        else:
            self.gate_proj_lora_a, self.gate_proj_lora_b = self.svd_init(expert.gate_proj)
            self.up_proj_lora_a, self.up_proj_lora_b = self.svd_init(expert.up_proj)
            self.down_proj_lora_a, self.down_proj_lora_b = self.svd_init(expert.down_proj)
        self.act_fn = ACT2FN[expert.config.hidden_act]

    def svd_init(self, linear: nn.Module):
        u, s, v = torch.linalg.svd(linear.weight, full_matrices=False)
        u = u[:, :self.rank]
        s = s[:self.rank]
        v = v[:self.rank, :]

        first = torch.nn.Parameter((u * s.unsqueeze(0)))
        second = torch.nn.Parameter(v)  # [r, in]
        return first, second  # [r, in] and [r, out]

    def forward(self, x):
        if self.rank is not None:
            gate_proj = F.linear(F.linear(x, self.gate_proj_lora_b), self.gate_proj_lora_a)
            up_proj = F.linear(F.linear(x, self.up_proj_lora_b), self.up_proj_lora_a)
            intermediate_states = self.act_fn(gate_proj) * up_proj
            down_proj = F.linear(F.linear(intermediate_states, self.down_proj_lora_b), self.down_proj_lora_a)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

    def set_requires_grad(self, requires_grad: bool):
        for param in self.parameters():
            param.requires_grad = requires_grad


class AvgDeepSeekExpert(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class MergedDeepseekMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, deepseek_moe):
        super().__init__()
        self.config = deepseek_moe.config
        self.num_experts_per_tok = deepseek_moe.num_experts_per_tok
        self.experts = deepseek_moe.experts
        self.gate = deepseek_moe.gate
        if deepseek_moe.config.n_shared_experts is not None:
            self.shared_experts = deepseek_moe.shared_experts

        self.avg_expert = None

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        avg_expert_weight = topk_weight.mean(dim=1, keepdim=True)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        if self.config.n_shared_experts is not None:
            if self.avg_expert is None:
                y = y + self.shared_experts(identity)
            else:
                y + self.shared_experts(
                    identity) + self.avg_expert(identity) * avg_expert_weight.view(orig_shape[0], -1, 1)
        return y

    # @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out = expert_out.mul(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache = expert_cache.scatter_reduce(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache
