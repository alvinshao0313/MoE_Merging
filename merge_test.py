import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.new_moe import MergedDeepseekMoE, AvgDeepSeekExpert, LoRAExpert
from models.deepseek.v1.modeling_deepseekv1 import DeepseekForCausalLM
from utils import data_utils, eval_utils, merge_utils, model_utils
import argparse
import torch.nn as nn
import copy
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the visible GPU device
parser = argparse.ArgumentParser(description="Evaluate DeepSeek MoE model")
parser.add_argument("--ppl_eval_batch_size", type=int, default=1, help="Batch size for evaluation")
parser.add_argument("--nsamples", type=int, default=128, help="Number of samples to use for merging")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
parser.add_argument("--rank", type=int, default=8, help="Rank for LoRA")
args = parser.parse_args()


model_name = "deepseek-ai/deepseek-moe-16b-base"
model = DeepseekForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model.seqlen = 2048  # Set the sequence length for the model
dataloader = data_utils.get_loaders(
    "wikitext2",
    seed=0,
    model=model_name,
    seqlen=model.seqlen,
    eval_mode=False,
)

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
use_cache = model.config.use_cache
model.config.use_cache = False
model.enable_input_require_grads()
model.model.embed_tokens = model.model.embed_tokens.to(dev)
model.model.norm = model.model.norm.to(dev)
if hasattr(model.model, "rotary_emb"):
    model.model.rotary_emb = model.model.rotary_emb.to(dev)

layers = model.model.layers  # model_utils.get_layers(model)
layers[0] = layers[0].to(dev)

org_inps = torch.zeros(
    (args.nsamples, model.seqlen, model.config.hidden_size), dtype=torch.float16, device=dev
)
cache = {"i": 0}

# catch the first layer input


class Catcher(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inp, **kwargs):
        org_inps[cache["i"]] = inp
        cache["i"] += 1
        cache["attention_mask"] = kwargs["attention_mask"]
        cache["position_ids"] = kwargs["position_ids"]
        raise ValueError


layers[0] = Catcher(layers[0])
with torch.no_grad():
    for batch in dataloader:
        if cache["i"] >= args.nsamples:
            break
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass

# move embedding layer and first layer to cpu
layers[0] = layers[0].module
layers[0] = layers[0].cpu()

model.model.embed_tokens = model.model.embed_tokens.cpu()
model.model.norm = model.model.norm.cpu()

merged_inps = copy.deepcopy(org_inps)
org_outputs = copy.deepcopy(org_inps)
attention_mask = cache["attention_mask"]

if attention_mask is not None:
    attention_mask_batch = attention_mask.repeat(
        args.batch_size, 1, 1, 1) if args.deactive_amp else attention_mask.repeat(args.batch_size, 1, 1, 1)  # .float()
else:
    attention_mask_batch = None

loss_func = torch.nn.MSELoss()
position_ids = cache["position_ids"]

for i in [1]:
    merging_layer = layers[i].to(dev)

    with torch.no_grad():
        for j in range(args.nsamples):
            org_outputs[j] = merging_layer(org_inps[j].unsqueeze(0), attention_mask=attention_mask,
                                           position_ids=position_ids)[0]

    experts_to_merge = merge_utils.get_experts(merging_layer, ['shared_experts'])
    average_params = merge_utils.average_merging(experts_to_merge, [])

    avg_expert = AvgDeepSeekExpert(
        config=model.config,
        hidden_size=model.config.hidden_size,
        intermediate_size=model.config.moe_intermediate_size
    )
    merge_utils.copy_params_to_model(average_params, avg_expert)
    for param_name, param in merging_layer.named_parameters():
        param.requires_grad = False
        param.grad = None

    layers[i].mlp = MergedDeepseekMoE(layers[i].mlp)
    layers[i].mlp.avg_expert = avg_expert.to(dev)
    merging_layer.mlp.experts = merge_utils.update_org_experts(merging_layer.mlp.experts, average_params)
    merging_layer.mlp.experts = merge_utils.init_lora_experts(merging_layer.mlp.experts, rank=args.rank)
    # merge_utils.set_lora_params_requires_grad(merging_layer, requires_grad=True)

    for param_name, param in merging_layer.named_parameters():
        if 'lora' in param_name:
            param.requires_grad = True

    if args.epochs > 0:
        # create optimizer
        optimizer = torch.optim.AdamW(
            [{"params": merge_utils.get_lora_params(merging_layer.mlp.experts), "lr": args.lr}])
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.epochs):
            loss_nan = False
            ep_loss = []
            for j in range(args.nsamples // args.batch_size):
                idx = j * args.batch_size
                with torch.cuda.amp.autocast():
                    merged_out = merging_layer(org_inps[idx:idx + args.batch_size], attention_mask=attention_mask,
                                               position_ids=position_ids)[0]
                    loss = nn.functional.mse_loss(merged_out, org_outputs[idx:idx + args.batch_size])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss.append(loss.detach().cpu())
                if not math.isfinite(loss.item()):
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()  # 清理显存缓存
                    loss_nan = True
                    break
            avg_loss = torch.stack(ep_loss).mean()
            print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss.item():.4f}")
            if loss_nan:
                print(f"Loss is NaN at epoch {epoch}, stopping training.")
                break


model.to('cuda:0')

for dataset in ['wikitext2']:
    testenc = data_utils.get_loaders(
        dataset,
        seed=0,
        model=model_name,
        seqlen=model.seqlen,
        eval_mode=True)
    dataset_ppl, total_tokens = eval_utils.ppl_evaluator(model, testenc, "cuda:0", args)
    print(f"Dataset: {dataset}, PPL: {dataset_ppl:.2f}, Total Tokens: {total_tokens}")
