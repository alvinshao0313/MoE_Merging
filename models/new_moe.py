import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
import math


class LoRAExpert(nn.Module):
    def __init__(self, expert: nn.Module, hidden_size=None, intermediate_size=None, rank=None):
        super().__init__()
        self.config = expert.config
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        assert rank is not None, "Rank must be specified for LoRAExpert"
        self.rank = rank
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


class MergedMixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, mixtral_moe):
        super().__init__()
        self.hidden_dim = mixtral_moe.hidden_size
        self.ffn_dim = mixtral_moe.intermediate_size
        self.num_experts = mixtral_moe.num_local_experts
        self.top_k = mixtral_moe.num_experts_per_tok

        # gating
        self.gate = mixtral_moe.gate

        self.experts = mixtral_moe.experts
        # Jitter parameters
        self.jitter_noise = mixtral_moe.router_jitter_noise

        self.avg_expert = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # if self.training and self.jitter_noise > 0:
        #     hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hitted = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero(as_tuple=True)[0].tolist()
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states = final_hidden_states.index_add(0, top_x, current_hidden_states.to(hidden_states.dtype))
            if self.avg_expert is not None:
                final_hidden_states = final_hidden_states.index_add(0, top_x, self.avg_expert(current_state))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
