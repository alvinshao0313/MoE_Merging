import torch
import torch.nn as nn
import math
import os
import logging
from tqdm import tqdm
from utils import model_utils


@torch.no_grad()
def evaluator(model, testenc, dev, args):

    model.eval()

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)

    layers[0] = layers[0].to(dev)

    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = testenc.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // model.seqlen  # The tail is truncated.
    input_ids = input_ids[:, :nsamples * model.seqlen].view(nsamples, model.seqlen).to(dev)  # (nsamples, seqlen)

    batch_size = 1  # args.bsz
    input_ids = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(input_ids)

    dtype = next(iter(model.parameters())).dtype
    # The input of the first decoder layer.
    inps = torch.zeros(
        (nbatches, batch_size, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    inps = [0] * nbatches
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])

    for i in range(nbatches):
        batch = input_ids[i]
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    position_ids = cache['position_ids']

    torch.cuda.empty_cache()
    outs = [0] * nbatches
    attention_mask = cache['attention_mask']

    for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
        layer = layers[i].to(dev)

        # Dump the layer input and output
        if args.capture_layer_io and args.layer_idx == i:
            captured_io = model_utils.capture_layer_io(model_utils.get_model_type(model), layer, inps)
            save_path = model_utils.get_layer_io_save_path(args)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(captured_io, save_path)
            logging.info(f'Dumped layer input and output to: {save_path}')

        for j in range(nbatches):
            bsz = inps[j].shape[0]
            outs[j] = layer(
                inps[j], attention_mask=attention_mask[0].repeat(
                    bsz, 1, 1, 1), position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)

    model.lm_head = model.lm_head.to(dev)
    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in range(nbatches):
        hidden_states = inps[i]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = input_ids[i][:, 1:]
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)
    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    model.config.use_cache = use_cache
    logging.info(f'\n{args.eval_dataset.upper()} PPL: {ppl.item():.3f}')
    return ppl.item()


@torch.no_grad()
def ppl_evaluator(model, testenc, dev, args):

    model.eval()
    total_tokens = 0

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = testenc.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // model.seqlen
    input_ids = input_ids[:, :nsamples * model.seqlen].view(nsamples, model.seqlen)  # (nsamples, seqlen)

    batch_size = args.ppl_eval_batch_size
    input_ids = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)]
    nlls = []

    # 初始化带有动态描述的tqdm
    pbar = tqdm(range(nsamples), desc="Evaluating PPL")

    for i in pbar:
        batch = input_ids[i].to(dev)
        total_tokens += batch.numel()
        outputs = model.model(batch)
        hidden_states = outputs[0]
        logits = model.lm_head(hidden_states)
        shift_logits = logits[:, :-1, :].to(model.lm_head.weight.device)
        shift_labels = input_ids[i][:, 1:].to(model.lm_head.weight.device)
        loss_fct = nn.CrossEntropyLoss().to(model.lm_head.weight.device)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float()
        if not math.isnan(neg_log_likelihood):
            nlls.append(neg_log_likelihood)

        # 动态更新tqdm描述中的PPL
        if nlls:
            current_ppl = torch.exp(torch.stack(nlls).mean())
            pbar.set_description(f"PPL: {current_ppl.item():.2f}")

    final_ppl = torch.exp(torch.stack(nlls).mean())
    model.config.use_cache = use_cache

    return final_ppl.item(), total_tokens
