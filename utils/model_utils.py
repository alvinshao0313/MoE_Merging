import torch
import typing
import transformers
import utils
import os
import logging
import models.deepseek.v1.modeling_deepseekv1 as deepseekv1
import models.deepseek.v2.modeling_deepseekv2 as deepseekv2


MIXTRAL_MODEL = transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM
MIXTRAL_LAYER = transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer
DEEPSEEKV1_MODEL = deepseekv1.DeepseekForCausalLM
DEEPSEEKV1_LAYER = deepseekv1.DeepseekDecoderLayer
DEEPSEEKV2_MODEL = deepseekv2.DeepseekV2ForCausalLM
DEEPSEEKV2_LAYER = deepseekv2.DeepseekV2DecoderLayer
QWEN2_MOE_MODEL = transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeForCausalLM
QWEN2_MOE_LAYER = transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeDecoderLayer
QWEN3_MOE_MODEL = transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeForCausalLM
QWEN3_MOE_LAYER = transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeDecoderLayer

supported_types = [
    MIXTRAL_MODEL,
    DEEPSEEKV1_MODEL,
    DEEPSEEKV2_MODEL,
    QWEN2_MOE_MODEL,
    QWEN3_MOE_MODEL,
]


def model_type_extractor(model):
    if isinstance(model, DEEPSEEKV1_MODEL):
        return DEEPSEEKV1_MODEL
    elif isinstance(model, DEEPSEEKV2_MODEL):
        return DEEPSEEKV2_MODEL
    elif isinstance(model, MIXTRAL_MODEL):
        return MIXTRAL_MODEL
    elif isinstance(model, QWEN2_MOE_MODEL):
        return QWEN2_MOE_MODEL
    elif isinstance(model, QWEN3_MOE_MODEL):
        return QWEN3_MOE_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')


def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization!
    pass


def get_rope_function_name(model):
    model_type = get_model_type(model)
    if model_type in supported_types:
        return "apply_rotary_pos_emb"
    else:
        raise NotImplementedError


def get_layers(model):
    model_type = get_model_type(model)
    if model_type in supported_types:
        return model.model.layers
    else:
        raise NotImplementedError


def get_mixtral(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.MixtralForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                            use_auth_token=hf_token,
                                                            low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_deepseekv1(model_name):
    model = deepseekv1.DeepseekForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                           low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_deepseekv2(model_name):
    model = deepseekv2.DeepseekV2ForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                             low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_qwen2_moe(model_name, hf_token):
    model = transformers.Qwen2MoeForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                             use_auth_token=hf_token,
                                                             low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_qwen3_moe(model_name, hf_token):
    model = QWEN3_MOE_MODEL.from_pretrained(model_name, torch_dtype='auto',
                                            use_auth_token=hf_token,
                                            low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_model(
    model_name, hf_token=None
):
    if 'Mixtral' in model_name:
        return get_mixtral(model_name, hf_token)
    elif 'deepseek-moe' in model_name:
        return get_deepseekv1(model_name)
    elif 'DeepSeek-V2' in model_name:
        return get_deepseekv2(model_name)
    elif 'Qwen2' in model_name:
        return get_qwen2_moe(model_name, hf_token)
    elif 'Qwen3' in model_name:
        return get_qwen3_moe(model_name, hf_token)
    else:
        raise ValueError(f'Unknown model {model_name}')


def get_model_type(model):
    if isinstance(model, MIXTRAL_MODEL):
        model_type = MIXTRAL_MODEL  # Mixtral model
    elif isinstance(model, DEEPSEEKV1_MODEL):
        model_type = DEEPSEEKV1_MODEL
    elif isinstance(model, QWEN2_MOE_MODEL):
        model_type = QWEN2_MOE_MODEL  # Qwen2 MoE model
    elif isinstance(model, QWEN3_MOE_MODEL):
        model_type = QWEN3_MOE_MODEL
    elif isinstance(model, DEEPSEEKV2_MODEL):
        model_type = DEEPSEEKV2_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')
    return model_type


def get_embeddings(model, model_type) -> list[torch.nn.Module]:
    if model_type in supported_types:
        return [model.model.embed_tokens]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_transformer_layers(model, model_type):
    if model_type in supported_types:
        return [layer for layer in model.model.layers]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_lm_head(model, model_type):
    if model_type in supported_types:
        return model.lm_head
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_pre_head_layernorm(model, model_type):
    if model_type == MIXTRAL_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm)
    elif model_type == DEEPSEEKV1_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm, deepseekv1.DeepseekRMSNorm)
    elif model_type == QWEN2_MOE_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeRMSNorm)
    elif model_type == QWEN3_MOE_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeRMSNorm)
    elif model_type == DEEPSEEKV2_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm, deepseekv2.DeepseekV2RMSNorm)
    else:
        raise ValueError(f'Unknown model type {model_type}')
    return pre_head_layernorm


def get_expert_bottleneck_size(model):
    model_type = get_model_type(model)
    if model_type in supported_types:
        return model.config.intermediate_size
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_moe_layer_range(model):
    model_type = get_model_type(model)
    if model_type in [DEEPSEEKV1_MODEL, DEEPSEEKV2_MODEL]:
        return [model.config.moe_layer_freq, model.config.num_hidden_layers]
    elif model_type in [MIXTRAL_MODEL, QWEN2_MOE_MODEL, QWEN3_MOE_MODEL]:
        return model.config.num_hidden_layers


def get_layer_io_save_path(args):
    return os.path.join(args.save_path, 'layer_io', f'{args.layer_idx:03d}.pt')


def capture_layer_io(model_type, layer, layer_input):
    def hook_factory(module_name, captured_vals, is_input):
        def hook(module, input, output):
            if is_input:
                captured_vals[module_name].append(input[0].detach().cpu())
            else:
                captured_vals[module_name].append(output.detach().cpu())
        return hook

    handles = []

    if model_type in supported_types:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'o_proj': [],
            'gate_proj': [],  # up_proj has the same input as gate_proj
            'down_proj': []
        }

        captured_outputs = {
            'v_proj': [],
        }

        for name in captured_inputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))

    else:
        raise ValueError(f'Unknown model type {model_type}')

    # Process each sequence in the batch one by one to avoid OOM.
    for seq_idx in range(layer_input.shape[0]):
        # Extract the current sequence across all dimensions.
        seq = layer_input[seq_idx:seq_idx + 1].to(utils.DEV)
        # Perform a forward pass for the current sequence.
        layer(seq)

    # After processing all sequences, concatenate the accumulated inputs for each sub-layer across the batch.
    for module_name in captured_inputs:
        captured_inputs[module_name] = torch.cat(captured_inputs[module_name], dim=0)
    for module_name in captured_outputs:
        captured_outputs[module_name] = torch.cat(captured_outputs[module_name], dim=0)

    # Cleanup.
    for h in handles:
        h.remove()

    return {
        'input': captured_inputs,
        'output': captured_outputs
    }
