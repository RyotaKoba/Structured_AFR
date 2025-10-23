import torch
import torch.nn as nn
from .layerwrapper import WrappedGPT

from transformers import AutoModelForCausalLM

import json
import os
from datetime import datetime
from .data import get_loaders
import math
from tqdm import tqdm
import sys
from .block_metrics import block_influence
import numpy as np
from .model import rm_modules, all_rm_modules

import torch.nn.utils.prune as prune

SCORE = None

class Pruner(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount):
        prune._validate_pruning_amount_init(amount)
        self.amount = amount
    
    def compute_mask(self, t, default_mask):
        tensor_size = t.nelement()
        print("amount:",self.amount)
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
        prune._validate_pruning_amount(nparams_toprune, tensor_size)

        print('number of parameters:', tensor_size)
        print('number of parameters to prune:', nparams_toprune)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        global SCORE
        print(SCORE.shape)
        if nparams_toprune != 0:
            topk = torch.topk(SCORE, k=nparams_toprune, largest=False)
            mask.view(-1)[topk.indices] = 0

        return mask

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    """
    Check the sparsity of the weights in different layers of the model.
    check_sparsity for llama3

    Args:
        model (nn.Module): The model to check.

    Returns:
        float: Ratio of the count of non-zero weights to total parameters in the model.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size

    # print(intermediate_size)
    # print(hidden_size)

    count = 0.0
    total_params = 0.0

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0

        for name in subset:
            # print(name)
            W = subset[name].weight.data
            sub_count += W.numel()
            # print(W.numel())

            count += W.numel()

            if name == 'self_attn.q_proj' or name == 'self_attn.o_proj':

                total_params += hidden_size * hidden_size
                sub_params += hidden_size * hidden_size

            elif name == 'self_attn.k_proj' or name == 'self_attn.v_proj':

                total_params += (hidden_size * hidden_size / 8)
                sub_params += (hidden_size * hidden_size / 8)

            else:

                total_params += hidden_size * intermediate_size
                sub_params += hidden_size * intermediate_size

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    print(total_params)
    model.config.use_cache = use_cache
    return float(count)/total_params

def check_sparsity_refer(model):#, save_path):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    count = 0 
    total_params = 0
    sparsity_data =[]
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            # total_params += W.numel()

            sub_count += (W==0).sum().item()
            # sub_params += W.numel()
            
            if name == 'self_attn.q_proj' or name == 'self_attn.o_proj':

                total_params += hidden_size * hidden_size
                sub_params += hidden_size * hidden_size

            elif name == 'self_attn.k_proj' or name == 'self_attn.v_proj':

                total_params += (hidden_size * hidden_size / 4)
                sub_params += (hidden_size * hidden_size / 4)
            else:

                total_params += hidden_size * intermediate_size
                sub_params += hidden_size * intermediate_size

        sparsity = float(sub_count) / sub_params if sub_params > 0 else 0.0
        sparsity_data.append({"Layer": i, "Sparsity": sparsity})
        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    # df = pd.DataFrame(sparsity_data)
    # df.to_csv(save_path, index=False)
    print(total_params)
    return float(count)/total_params, float(count)



def prepare_calibration_input(model, dataloader, device):
    """
    Prepare inputs for model calibration.

    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded.

    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        attention_mask (torch.Tensor): Attention mask tensor.
        position_ids (torch.Tensor): Position IDs tensor.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype

    inps = torch.zeros((2048, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)  # 2048 is the upper limit.

    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
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

    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def compress(layer, mlp_mask, device):#形状を変更

    mlp_mask = mlp_mask.to(device)

    layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
    layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]

    # Update output dimensions of up and gate projections based on the mlp mask
    layer.mlp.up_proj.out_features = mlp_mask.sum().item()
    layer.mlp.gate_proj.out_features = mlp_mask.sum().item()

    output_weight = layer.mlp.down_proj.weight.data
    layer.mlp.intermediate_size = mlp_mask.sum().item()
    print(layer.mlp.intermediate_size)

    # Prune the down projection weight
    output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]

    # Assign the pruned weights
    layer.mlp.down_proj.weight.data = output_weight

    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()

# def compress(layer, mlp_mask, device): #形状は維持　０置換
#     mlp_mask = mlp_mask.to(device)
    
#     # 重みを直接削除する代わりに、マスクを使用して不要な重みを0に設定
#     # up_projの処理
#     masked_up_weight = torch.zeros_like(layer.mlp.up_proj.weight.data)
#     masked_up_weight[torch.where(mlp_mask)[0]] = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
#     layer.mlp.up_proj.weight.data = masked_up_weight
    
#     # gate_projの処理
#     masked_gate_weight = torch.zeros_like(layer.mlp.gate_proj.weight.data)
#     masked_gate_weight[torch.where(mlp_mask)[0]] = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]
#     layer.mlp.gate_proj.weight.data = masked_gate_weight
    
#     # down_projの処理
#     masked_down_weight = torch.zeros_like(layer.mlp.down_proj.weight.data)
#     masked_down_weight[:, torch.where(mlp_mask)[0]] = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]
#     layer.mlp.down_proj.weight.data = masked_down_weight
    
#     # 元のサイズ情報を維持（テンソル形状は変更せず）
#     # layer.mlp.intermediate_size = 14336 # 元のサイズを維持
    
#     # スパース性を記録（オプション）
#     effective_neurons = mlp_mask.sum().item()
#     print(f"有効なニューロン数: {effective_neurons}/{len(mlp_mask)}")
    
#     # メモリの解放
#     torch.cuda.empty_cache()

# for flap
def compress_bias(layer, mlp_mask, mlp_mean_inp, device):

    bias = True

    layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
    layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]

    # Update output dimensions of up and gate projections based on the mlp mask
    layer.mlp.up_proj.out_features = mlp_mask.sum().item()
    layer.mlp.gate_proj.out_features = mlp_mask.sum().item()

    output_weight = layer.mlp.down_proj.weight.data
    layer.mlp.intermediate_size = mlp_mask.sum().item()
    if bias:
        # Add the additional bias to compensate for the loss
        output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)

    # Prune the down projection weight
    output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]

    if bias:
        # Re-initialize the Linear layer with new shape and bias
        layer.mlp.down_proj.in_features = mlp_mask.sum().item()
        # layer.mlp.down_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
        layer.mlp.down_proj.bias.data = output_bias

    # Assign the pruned weights
    layer.mlp.down_proj.weight.data = output_weight

    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()




def prune_cfsp(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    our method
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers


    mlp_metric_list = []
    mlp_mask = []

    layer_importances = []

    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = {}
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        layer_importance = 0.0

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

                if args.global_metrics == 'angular':
                    layer_importance += block_influence(inps[j].unsqueeze(0), outs[j].unsqueeze(0), metrics='angular').sum().cpu().item()
                elif args.global_metrics == 'cosine':
                    layer_importance += block_influence(inps[j].unsqueeze(0), outs[j].unsqueeze(0), metrics='cosine').sum().cpu().item()
                elif args.global_metrics == 'mse':
                    layer_importance += block_influence(inps[j].unsqueeze(0), outs[j].unsqueeze(0), metrics='mse').sum().cpu().item()
                elif args.global_metrics == 'mae':
                    layer_importance += block_influence(inps[j].unsqueeze(0), outs[j].unsqueeze(0), metrics='mae').sum().cpu().item()
                else:
                    layer_importance += 100
            print("korenani, j:",j)
            



        layer_importances.append(layer_importance)
        for h in handles:
            h.remove()

        for name in subset:
            if args.local_metrics == "wanda_base":
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            elif args.local_metrics == "mag_base":
                W_metric = torch.norm(subset[name].weight.data, dim=0)

            elif args.local_metrics == "one_a":
                W = subset[name].weight.data
                W_metric = (torch.abs(W)/torch.sum(torch.abs(W), dim=0)) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**0.5

            elif args.local_metrics == "three_w_one_a":
                W = subset[name].weight.data
                # W_down: torch.Size([4096, 14336])
                # W_up: torch.Size([14336, 4096])
                # W_gate: torch.Size([14336, 4096])
                W_up =  find_layers(layer)['mlp.up_proj'].weight.data
                W_gate = find_layers(layer)['mlp.gate_proj'].weight.data
                W_up = W_up.t()
                W_gate = W_gate.t()
                W_metric = ((torch.abs(W)/torch.sum(torch.abs(W), dim=0)) +
                            (torch.abs(W_up)/torch.sum(torch.abs(W_up), dim=0))+
                            (torch.abs(W_gate)/torch.sum(torch.abs(W_gate), dim=0))) \
                            * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.c

            elif args.local_metrics == "three_w_one_wa":
                W = subset[name].weight.data
                # W_down: torch.Size([4096, 14336])
                # W_up: torch.Size([14336, 4096])
                # W_gate: torch.Size([14336, 4096])
                W_under = (torch.abs(W) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.b)
                W_up =  find_layers(layer)['mlp.up_proj'].weight.data
                W_gate = find_layers(layer)['mlp.gate_proj'].weight.data
                W_up = W_up.t()
                W_gate = W_gate.t()
                W_metric = ((torch.abs(W_under)/torch.sum(torch.abs(W_under), dim=0)) +
                            (torch.abs(W_up)/torch.sum(torch.abs(W_up), dim=0))+
                            (torch.abs(W_gate)/torch.sum(torch.abs(W_gate), dim=0))) \
                            * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.c

            elif args.local_metrics == "one_wa":
                W = subset[name].weight.data
                W_under = (torch.abs(W) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.b)
                W_metric = (torch.abs(W_under)/torch.sum(torch.abs(W_under), dim=0)) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.c

            # print("W_metric: ", W_metric.shape)
            W_metric = W_metric.mean(axis=0)
            # print("W_metric: ", W_metric.shape)
            mlp_metric_list.append(W_metric.cpu())

            wrapped_layers[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps # the pruned output as input to the next layer

        torch.cuda.empty_cache()


    layer_importances_sorted = sorted(enumerate(layer_importances), key=lambda x: x[1], reverse=True)

    for i in range(len(layer_importances_sorted)):
        index2 = layer_importances_sorted[i][0]
        number2 = layer_importances_sorted[i][1]
        print(f"layer: {index2} , importance: {number2} ")

    print(f"{args.global_metrics} layer_importances_sorted: {layer_importances_sorted}")


    def sigmoid(x):
        return 1 / (1 + np.exp(-x*args.a))

    layer_importances_mid = sum(layer_importances) / len(layer_importances)

    layer_importances = [(i-layer_importances_mid)/1e4 for i in layer_importances]
    layer_importances = [sigmoid(i) for i in layer_importances]


    avg = sum(layer_importances) / len(layer_importances)
    max_score = max(layer_importances)
    if max_score / avg * (1-args.pruning_ratio) >= 1:
        #
        scale_factor = (avg * (1 / (1-args.pruning_ratio) - 1)) /  (max_score - avg) / 1.05
        for i in range(len(layer_importances)):
            if layer_importances[i] > avg:
                layer_importances[i] = avg + (layer_importances[i] - avg) * scale_factor
            else:
                layer_importances[i] = avg - (avg - layer_importances[i]) * scale_factor
        avg = sum(layer_importances) / len(layer_importances)

    print("mlp_metric_list:", len(mlp_metric_list))
    print("mlp_metric_list:", len(mlp_metric_list[0]))
    mlp_metric = torch.stack(mlp_metric_list)
    print("mlp_metric: ", mlp_metric.shape)

    sorted_mlp_metric, _ = torch.sort(mlp_metric, descending=True)
    # print(sorted_mlp_metric.shape)

    every_pruning_ratios = [i/avg*(1-args.pruning_ratio) for i in layer_importances]
    print(f"every_pruning_ratios: {every_pruning_ratios}")



    if args.cuda_friendly:
        thresholds = torch.tensor([
            sorted_mlp_metric[i][int(((sorted_mlp_metric.shape[1]*every_pruning_ratios[i])+64)/128)*128-1]
                                   for i in range(len(every_pruning_ratios))
                                   ])
        print(f"thresholds: {thresholds}")

    else:
        thresholds = torch.tensor([sorted_mlp_metric[i][int(sorted_mlp_metric.shape[1]*every_pruning_ratios[i])] for i in range(len(every_pruning_ratios))])
        print(f"thresholds: {thresholds}")


    if len(every_pruning_ratios) == len(layers):
        print("そのまま使える〜〜")

    mlp_mask = (mlp_metric.t() >= thresholds).t()
    print(mlp_mask.shape)
    print("mlp_mask: ", mlp_mask)

    print('*'*30)
    for idx in range(len(layers)):
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            compress(model.model.layers[idx], mlp_mask[idx], model.hf_device_map[f"model.layers.{idx}"])
        else:
            compress(model.model.layers[idx], mlp_mask[idx], device)

    print('*'*30)
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def snip(args, model, tokenizer, device):
    device = [i + 1 for i in range(device - 1)]
    # model = nn.DataParallel(model, device_ids=device).to('cuda:1')
    print("loading calibdation data")
    dataloader, _ = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    if args.all:
        rm_module = all_rm_modules(model)
    else:
        rm_module = rm_modules(model)

    rm_weights = [module.weight for module, _ in rm_module]

    with torch.no_grad():
        grads = [torch.zeros_like(w) for w in rm_weights]
    
    for i, (inp, tar) in enumerate(dataloader):
        outputs = model(inp)
        outputs = outputs.logits
        outputs = outputs.reshape(-1, outputs.shape[-1])
        tar = tar.reshape(-1)
        loss = nn.CrossEntropyLoss()(outputs, tar)
        grads = list(torch.autograd.grad(loss, rm_weights))
        break

    with torch.no_grad():
        score = [(weight.cpu() * grad.cpu()).view(-1).abs() for weight, grad in zip(rm_weights, grads)]
        print("score: ", len(score))
        score = torch.cat(score)
    print("score: ", len(score))
    
    model.zero_grad()
    return score

def structured_snip(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    
    print("loading calibdation data")
    dataloader, _ = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    # inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    # print(f"calied inp:{inps[0].unsqueeze(0).shape}")
    
    rm_module = rm_modules(model)
    
    rm_weights = [module.weight for module, _ in rm_module]
    
    # input_batch = torch.stack([item[0] for item in dataloader]).squeeze(1)  # バッチの最初の要素を取得

#########################################################
    for i, (inp, tar) in enumerate(dataloader):
        print(f"no calied inp:{inp.shape}")
        # inp = inp.to(device)
        # tar = tar.to(device)
        outputs = model(inp)
        outputs = outputs.logits
        outputs = outputs.reshape(-1, outputs.shape[-1])
        tar = tar.reshape(-1)
        loss = nn.CrossEntropyLoss()(outputs, tar)
        grads = list(torch.autograd.grad(loss, rm_weights))
        break

    layers = model.model.layers

    mlp_metric_list = []
    mlp_mask = []

    for i in tqdm(range(len(layers)), desc="Processing layers"):
        W_down = rm_weights[i+64] * grads[i+64]
        W_up = (rm_weights[i+32] * grads[i+32]).t()
        W_gate = (rm_weights[i] * grads[i]).t()
        W_metric = W_down + W_up + W_gate
        W_metric = W_metric.mean(axis=0)
        mlp_metric_list.append(W_metric)
        print("W_metric:",W_metric)
    print("mlp_metric_list: ", mlp_metric_list[0])
#######################################################################
    # layers = model.model.layers
    # mlp_metric_list = []
    # mlp_mask = []


    # for i, (inp, tar) in tqdm(enumerate(dataloader), desc="Processing batches", total=args.nsamples):
    #     print(f"no calied inp:{inp.shape}")
    #     # inp = inp.to(device)
    #     # tar = tar.to(device)
    #     outputs = model(inp)
    #     outputs = outputs.logits
    #     outputs = outputs.reshape(-1, outputs.shape[-1])
    #     tar = tar.reshape(-1)
    #     loss = nn.CrossEntropyLoss()(outputs, tar)
    #     grads = list(torch.autograd.grad(loss, rm_weights))  # retain_graph=True to keep the graph for multiple iterations
    #     print("grads: ", len(grads[0]))
    #     # break
    #     # with torch.no_grad():
    #     for j in tqdm(range(len(layers)), desc="Processing layers"):
    #         W_down = rm_weights[j+64] * grads[j+64]
    #         W_up = (rm_weights[j+32] * grads[j+32]).t()
    #         W_gate = (rm_weights[j] * grads[j]).t()
    #         W_metric = W_down + W_up + W_gate
    #         W_metric = W_metric.mean(axis=0)
    #         if i == 0 :
    #             print("W_metric:",W_metric)
    #             mlp_metric_list.append(W_metric)
    #         else:
    #             mlp_metric_list[j] += W_metric.detach()
    #     print("mlp_metric_list: ", len(mlp_metric_list[0]))
    #     print("mlp_metric_list: ", mlp_metric_list[0])
    #     del grads
    #     model.zero_grad()

    #     torch.cuda.empty_cache()

######################################################

    mlp_metric = torch.stack(mlp_metric_list)
    print("mlp_metric: ", mlp_metric.shape)

    sorted_mlp_metric, _ = torch.sort(mlp_metric, descending=True)#flat
    # sorted_mlp_metric, _ = torch.sort(mlp_metric.view(-1), descending=True)#uneven
    # limit = sorted_mlp_metric[int(sorted_mlp_metric.shape[0]*(1-args.pruning_ratio))]#uneven
    # thresholds = torch.tensor([limit for i in range(len(layers))])#uneven

    thresholds = torch.tensor([sorted_mlp_metric[i][int(sorted_mlp_metric.shape[1]*(1-args.pruning_ratio))] for i in range(len(layers))])#flat
    print(f"thresholds: {thresholds}")
    print(thresholds.shape)

    mlp_mask = (mlp_metric.t() >= thresholds).t()
    # mlp_mask = (mlp_metric.t() <= thresholds).t() #reverse
    
    print(mlp_mask)
    print(mlp_mask.shape)

    print('*'*30)
    for idx in range(len(layers)):
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            compress(model.model.layers[idx], mlp_mask[idx], model.hf_device_map[f"model.layers.{idx}"])
        else:
            compress(model.model.layers[idx], mlp_mask[idx], device)

    print('*'*30)
    model.zero_grad()  # 勾配をリセット

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

loss = torch.zeros(1)
def ReFer_L1(args, model,tokenizer, device):
    device = [i + 1 for i in range(device - 1)]
    
    print("loading calibdation data")
    dataloader, _ = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    if args.all:
        rm_module = all_rm_modules(model)
    else:
        rm_module = rm_modules(model)

    rm_weights = [module.weight for module, _ in rm_module]


    
    global loss
    def store_feature(module, input, output):
        global loss
        # if 'LlamaMLP' in module.__class__.__name__:
            # loss = loss + output.abs().sum().to('cuda:1')
        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
            output = output.hidden_states
        elif isinstance(output, tuple):
            output = output[0]

        if output is None:
            return
        loss = loss + output.abs().sum()

    for _, module in model.named_modules():
        module.register_forward_hook(store_feature)
    # model = nn.DataParallel(model, device_ids=device)
    
    for i, (inputs, targets) in enumerate(dataloader):
        # inputs, targets = inputs.cuda(0), targets.cuda(0) # 1バッチ分のデータだけ取り出しておく
        outputs = model(inputs)  # フォワードパス
        grads = list(torch.autograd.grad(loss, rm_weights))
        break

    # for batch in dataloader:
    #     outputs = model(batch[0])
    #     print(loss.shape)
    #     print(rm_weights[0].shape)

    #     grads = list(torch.autograd.grad(loss, rm_weights))
    #     # print(grads)
    #     break
    
    with torch.no_grad():
        score=[(weight.cpu() * grad.cpu()).view(-1).abs() for weight, grad in zip(rm_weights, grads)]
        score = torch.cat(score)
    
    model.zero_grad()
    # loss = torch.zeros(1).to('cuda:1')

    return score
    
P_SVD_loss = torch.zeros(1)
def AFR(args, model, tokenizer, device):
    device = [i for i in range(device)]
    print("loading calibdation data")
    dataloader, _ = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    global P_SVD_loss 
    P_SVD_loss = torch.zeros(1, requires_grad=True)
    # 特徴空間の損失を取る関数．hookで呼ばれる．
    def store_feature(module, input, output):
        global P_SVD_loss
        
        # if 'LlamaMLP' in module.__class__.__name__:


        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
            print("here!!!!!!!!!!!!!!")
            print("module: ", module.__class__.__name__)
            return
        elif hasattr(output, 'hidden_states'):
            output = output.hidden_states
            print("aohere!!!!!!!!!!!!!!")
            print("module: ", module.__class__.__name__)
            return
        elif isinstance(output, tuple):
            output = output[0]
       

        if output is None:
            print("output is None")
            return
        
        # # 出力テンソルの特異値分解を実行
        # print("before output: ", output.shape)
        # output = output.reshape(output.size(0), -1)  # バッチサイズ次元以外をフラットにする
        # print("after output: ", output.shape)
        # U, S, Vh = torch.svd(output)  # SVDを計算
        # # print("S: ", S.shape)
        # # print("S: ", S)1sannpuru
        # # 特異値の平均を計算してP_SVD_lossに加算
        # singular_value_mean = S.mean()
        # # print("singular_value_mean: ", singular_value_mean)
        # # P_SVD_loss += singular_value_mean
        # P_SVD_loss = P_SVD_loss + singular_value_mean

         # [1, seq, dim] → [seq, dim] - これが重要！
        if output.dim() == 3 and output.size(0) == 1:
            output = output.squeeze(0)  # [1024, 4096]
            
        print(f"True SVD shape: {output.shape}")
        
        # 真の特異値分解
        U, S, Vh = torch.svd(output)  # [1024, 4096]のSVD
        print(f"Singular values shape: {S.shape}")  # [1024]
        print(f"Number of singular values: {len(S)}")
        
        # 特異値の平均（複数の特異値から）
        singular_value_mean = S.mean()
        P_SVD_loss = P_SVD_loss + singular_value_mean
        
        print(f"True SVD mean: {singular_value_mean.item():.6f}")

    # 全てのモジュールに対してhookをかける．
    hooks = []
    for name, module in model.named_modules():
        
        hook = module.register_forward_hook(store_feature)
        hooks.append(hook) 

    # model = nn.DataParallel(model, device_ids=device).to('cuda:0')

    if args.all:
        rm_module = all_rm_modules(model)
    else:
        rm_module = rm_modules(model)


    rm_weights = [module.weight for module, _ in rm_module]  # FOが枝刈りを担当する重み

    # 勾配初期化
    with torch.no_grad():
        fo_grads = [torch.zeros_like(w) for w in rm_weights]
        snip_grads = [torch.zeros_like(w) for w in rm_weights]

    # 1バッチのみ処理して勾配を計算
    for i, (inputs, targets) in enumerate(dataloader):
        # inputs, targets = inputs.cuda(0), targets.cuda(0) # 1バッチ分のデータだけ取り出しておく
        outputs = model(inputs)  # フォワードパス
        break
    print("P_SVD_loss: ", P_SVD_loss)
    print("P_SVD_loss: ", P_SVD_loss.shape)
    fo_grads = list(torch.autograd.grad(P_SVD_loss, rm_weights))  # FOの勾配を計算
    for hook in hooks:
        hook.remove() # メモリ消費えぐいのでhookを外す
    
    P_SVD_loss = torch.zeros(1)
    
    # FOスコアの計算
    with torch.no_grad():
        fo_score = [(weight.cpu() * grad.cpu()).view(-1).abs() for weight, grad in zip(rm_weights, fo_grads)]
        fo_score = torch.cat(fo_score)

    outputs = model(inputs)
    outputs = outputs.logits
    outputs = outputs.reshape(-1, outputs.shape[-1])
    targets = targets.reshape(-1)
    loss = nn.CrossEntropyLoss()(outputs, targets) # CE Loss
    snip_grads = list(torch.autograd.grad(loss, rm_weights)) #SNIPの勾配を計算
    
    # SNIPスコアの計算
    with torch.no_grad():
        snip_score = [(weight.cpu() * grad.cpu()).view(-1).abs() for weight, grad in zip(rm_weights, snip_grads)]
        snip_score = torch.cat(snip_score)
    
    # fo_score と snip_score を標準化
    fo_score_standardized = (fo_score - fo_score.mean()) / fo_score.std()
    snip_score_standardized = (snip_score - snip_score.mean()) / snip_score.std()
    
    # FOスコアとSNIPスコアの結合
    score = fo_score_standardized + snip_score_standardized
    # score = snip_score_standardized
    # score = fo_score_standardized
    

    
    model.zero_grad()  # 勾配をリセット
    del P_SVD_loss  # グローバル変数をリセット
    P_SVD_loss = torch.zeros(1)  # グローバル変数の再初期化
    return score  # スコアを返す

SVD_loss = torch.zeros(1)
def ReFer_SVD(args, model, tokenizer, device):
    print("Start ReFer_SVD")
    device = [i + 1 for i in range(device - 1)]
    print("loading calibdation data")
    dataloader, _ = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")

    global SVD_loss
    # 特異値分解のフックを定義
    def store_feature(module, input, output):
        global SVD_loss
        # if 'LlamaMLP' in module.__class__.__name__:

        if hasattr(output, 'last_hidden_state'):
        # transformers の出力オブジェクトから hidden_states を取得
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
        # hidden_states が利用可能な場合
            output = output.hidden_states
        elif isinstance(output, tuple):
        # タプルの場合は最初の要素を使用
            output = output[0]

        if output is None:
            return


        # 出力テンソルの特異値分解を実行
        output = output.reshape(output.size(0), -1)  # バッチサイズ次元以外をフラットにする
        U, S, Vh = torch.svd(output)  # SVDを計算
        # 特異値の平均を計算してSVD_lossに加算
        singular_value_mean = S.mean()
        # SVD_loss += singular_value_mean
        SVD_loss = SVD_loss + singular_value_mean
        
    # モデル内の各モジュールにフックを追加
    for _, module in model.named_modules():
        module.register_forward_hook(store_feature)

    # model = nn.DataParallel(model, device_ids=device).to('cuda:1')

    if args.all:
        rm_module = all_rm_modules(model)
    else:
        rm_module = rm_modules(model)

    rm_weights = [module.weight for module, _ in rm_module]

    with torch.no_grad():
        grads = [torch.zeros_like(w) for w in rm_weights]
    
    for i, (inputs, targets) in enumerate(dataloader):
        # inputs, targets = inputs.cuda(1), targets.cuda(1)
        outputs = model(inputs)

        grads = list(torch.autograd.grad(SVD_loss, rm_weights))
        break

    with torch.no_grad():
        score = [(weight.cpu() * grad.cpu()).view(-1).abs() for weight, grad in zip(rm_weights, grads)]
        score = torch.cat(score)
    
    model.zero_grad()
    #del loss
    loss = torch.zeros(1)

    return score

SVD_loss = torch.zeros(1)
def Structured_ReFer_SVD(args, model, tokenizer, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print("Start ReFer_SVD")
    # device = [i + 1 for i in range(device - 1)]
    print("loading calibdation data")
    dataloader= get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")

    rm_module = rm_modules(model)
    rm_weights = [module.weight for module, _ in rm_module]

    class WeightScoreLogger:
        def __init__(self, save_dir="weight_scores"):
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
            
            # FOとSNIPのスコアを分けて保存
            self.fo_weight_scores = []
            self.snip_weight_scores = []
            
            # メタデータ
            self.metadata = {
                'timestamp': datetime.now().isoformat(),
                'layers_processed': 0,
                'score_shape': None
            }
        
        def save_fo_layer_scores(self, layer_idx, W_metric):
            """FOの重み単位スコアを保存"""
            # [hidden_dim, intermediate_size] の形状で保存
            self.fo_weight_scores.append({
                'layer_idx': layer_idx,
                'W_metric': W_metric.cpu().clone(),
                'shape': list(W_metric.shape)
            })
            
        def save_snip_layer_scores(self, layer_idx, W_metric):
            """SNIPの重み単位スコアを保存"""
            self.snip_weight_scores.append({
                'layer_idx': layer_idx,
                'W_metric': W_metric.cpu().clone(),
                'shape': list(W_metric.shape)
            })
        
        def save_to_files(self):
            """ファイルに保存"""
            # FOスコア保存
            torch.save(self.fo_weight_scores, 
                    os.path.join(self.save_dir, 'fo_weight_scores.pt'))
            
            # SNIPスコア保存
            torch.save(self.snip_weight_scores, 
                    os.path.join(self.save_dir, 'snip_weight_scores.pt'))
            
            # メタデータ保存
            with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            print(f"Saved weight scores to {self.save_dir}")
            print(f"FO layers: {len(self.fo_weight_scores)}")
            print(f"SNIP layers: {len(self.snip_weight_scores)}")

    def calculate_neuron_score_v2(W_metric):
        """
        方法2: Signal-to-Noise比的なアプローチ
        平均の絶対値 / (標準偏差 + epsilon)
        """

        """トリム平均を使用してニューロンスコアを計算"""
        # 上位5%と下位5%を除外する設定
        trim_percent = 2
        
        # 各列をソートして上位・下位を除外
        sorted_W, _ = torch.sort(W_metric, dim=0)
        n_rows = W_metric.shape[0]  # 4096
        
        # 除外する要素数を計算
        trim_count = int(n_rows * trim_percent / 100)
        
        # 中央部分を抽出して平均を計算
        trimmed_W = sorted_W[trim_count:-trim_count, :]
        mean_scores = trimmed_W.mean(axis=0)
        std_scores = trimmed_W.std(axis=0)
        snr_scores = torch.abs(mean_scores) / (std_scores + 1e-8)
        return snr_scores

    global SVD_loss
    SVD_loss = torch.zeros(1, requires_grad=True, dtype=torch.float32).to("cpu")


    logger = WeightScoreLogger(save_dir=f"weight_scores_{args.pruning_ratio}")

    # 特異値分解のフックを定義
    def store_feature(module, input, output):
        global SVD_loss
        # if 'LlamaMLP' in module.__class__.__name__:

        if hasattr(output, 'last_hidden_state'):
        # transformers の出力オブジェクトから hidden_states を取得
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
        # hidden_states が利用可能な場合
            output = output.hidden_states
        elif isinstance(output, tuple):
        # タプルの場合は最初の要素を使用
            output = output[0]

        if output is None:
            return

        # 出力テンソルの特異値分解を実行
        output = output.reshape(output.size(0), -1).to(dtype=torch.float32)  # バッチサイズ次元以外をフラットにする
        U, S, Vh = torch.svd(output)  # SVDを計算
        # 特異値の平均を計算してSVD_lossに加算
        singular_value_mean = S.mean().to("cpu",dtype=torch.float32)
        # SVD_loss += singular_value_mean
        SVD_loss = SVD_loss + singular_value_mean
        
    # モデル内の各モジュールにフックを追加
    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(store_feature)
        hooks.append(hook) 

    with torch.no_grad():
        grads = [torch.zeros_like(w) for w in rm_weights]
    
    for i, (inputs, targets) in enumerate(dataloader):
        # inputs, targets = inputs.cuda(1), targets.cuda(1)
        outputs = model(inputs)
        break
    print("P_SVD_loss contains nan:", torch.isnan(SVD_loss).any().item())
    print("P_SVD_loss contains inf:", torch.isinf(SVD_loss).any().item())
    print("P_SVD_loss tensor dtype:", SVD_loss.dtype)
    print("P_SVD_loss shape:", SVD_loss.shape)
    grads = list(torch.autograd.grad(SVD_loss, rm_weights))
    for i, grad in enumerate(grads):
        has_inf = torch.isinf(grad).any().item()
        has_nan = torch.isnan(grad).any().item()
        if has_inf or has_nan:
            print(f"fo_grads[{i}] contains inf or nan")
        else:
            print(f"fo_grads[{i}] is clean")
    for hook in hooks:
        hook.remove() # メモリ消費えぐいのでhookを外す
    P_SVD_loss = torch.zeros(1)
    layers = model.model.layers
    mlp_metric_list = []
    mlp_mask = []
    with torch.no_grad():
        for i in tqdm(range(len(layers)),desc="Processing layers"):
            W_down = rm_weights[i+64] * grads[i+64]
            W_up = (rm_weights[i+32] * grads[i+32]).t()
            W_gate = (rm_weights[i] * grads[i]).t()
            W_metric = W_down + W_up + W_gate
            print(f"SVD_W_metric {i} have NaN:", torch.isnan(W_metric).any().item())
            print(f"SVD_W_metric {i} have inf:", torch.isinf(W_metric).any().item())
            logger.save_fo_layer_scores(i, W_metric)
            W_metric = calculate_neuron_score_v2(W_metric)
            # W_metric = W_metric.mean(axis=0)
            mlp_metric_list.append(W_metric.cpu())
    mlp_metric = torch.stack(mlp_metric_list)
    print("score contains nan:", torch.isnan(mlp_metric).any().item())
    print("score contains inf:", torch.isinf(mlp_metric).any().item())
    print("score: ", mlp_metric.shape)
    print("score: ", mlp_metric)
    sorted_mlp_metric, _ = torch.sort(mlp_metric, descending=True)#flat
    # sorted_mlp_metric, _ = torch.sort(mlp_metric.view(-1), descending=True)#uneven
    # limit = sorted_mlp_metric[int(sorted_mlp_metric.shape[0]*(1-args.pruning_ratio))]#uneven
    # thresholds = torch.tensor([limit for i in range(len(layers))])#uneven
    thresholds = torch.tensor([sorted_mlp_metric[i][int(sorted_mlp_metric.shape[1]*(1-args.pruning_ratio))] for i in range(len(layers))])#flat
    mlp_mask = (mlp_metric.t() >= thresholds).t()
    
    print('*'*30)
    for idx in range(len(layers)):
        compress(model.model.layers[idx], mlp_mask[idx], device)

    print('*'*30)
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    logger.save_to_files()


loss = torch.zeros(1)
def Structured_ReFer_L1(args, model,tokenizer, device):
    # device = [i + 1 for i in range(device - 1)]
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    print("loading calibdation data")
    dataloader, _ = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    rm_module = rm_modules(model)
    rm_weights = [module.weight for module, _ in rm_module]

    
    global loss
    def store_feature(module, input, output):
        global loss
        # if 'LlamaMLP' in module.__class__.__name__:
            # loss = loss + output.abs().sum().to('cuda:1')
        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
            output = output.hidden_states
        elif isinstance(output, tuple):
            output = output[0]

        if output is None:
            return
        loss = loss + output.abs().sum()

    for _, module in model.named_modules():
        module.register_forward_hook(store_feature)
    
    for i, (inputs, targets) in enumerate(dataloader):
        # inputs, targets = inputs.cuda(0), targets.cuda(0) # 1バッチ分のデータだけ取り出しておく
        outputs = model(inputs)  # フォワードパス
        grads = list(torch.autograd.grad(loss, rm_weights))
        break
    layers = model.model.layers
    mlp_metric_list = []
    mlp_mask = []
    with torch.no_grad():
        for i in tqdm(range(len(layers)),desc="Processing layers"):
            W_down = rm_weights[i+64] * grads[i+64]
            W_up = (rm_weights[i+32] * grads[i+32]).t()
            W_gate = (rm_weights[i] * grads[i]).t()
            W_metric = W_down + W_up + W_gate
            W_metric = W_metric.mean(axis=0)
            mlp_metric_list.append(W_metric.cpu())

    mlp_metric = torch.stack(mlp_metric_list)
    sorted_mlp_metric, _ = torch.sort(mlp_metric, descending=True)#flat
    # sorted_mlp_metric, _ = torch.sort(mlp_metric.view(-1), descending=True)#uneven
    # limit = sorted_mlp_metric[int(sorted_mlp_metric.shape[0]*(1-args.pruning_ratio))]#uneven
    # thresholds = torch.tensor([limit for i in range(len(layers))])#uneven
    thresholds = torch.tensor([sorted_mlp_metric[i][int(sorted_mlp_metric.shape[1]*(1-args.pruning_ratio))] for i in range(len(layers))])#flat
    mlp_mask = (mlp_metric.t() >= thresholds).t()
    

    print('*'*30)
    for idx in range(len(layers)):
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            compress(model.model.layers[idx], mlp_mask[idx], model.hf_device_map[f"model.layers.{idx}"])
        else:
            compress(model.model.layers[idx], mlp_mask[idx], device)

    print('*'*30)
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    model.zero_grad()


P_SVD_loss = torch.zeros(1)
def Structured_AFR(args, model, tokenizer, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    # device = [i for i in range(device)]
    print("loading calibdation data")
    dataloader = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    def calculate_neuron_score_v2(W_metric):
        """
        方法2: Signal-to-Noise比的なアプローチ
        平均の絶対値 / (標準偏差 + epsilon)
        """

        """トリム平均を使用してニューロンスコアを計算"""
        # 上位5%と下位5%を除外する設定
        trim_percent = 2
        
        # 各列をソートして上位・下位を除外
        sorted_W, _ = torch.sort(W_metric, dim=0)
        n_rows = W_metric.shape[0]  # 4096
        
        # 除外する要素数を計算
        trim_count = int(n_rows * trim_percent / 100)
        
        # 中央部分を抽出して平均を計算
        trimmed_W = sorted_W[trim_count:-trim_count, :]
        # trimmed_W = sorted_W[trim_count:, :]
        mean_scores = trimmed_W.mean(axis=0)
        std_scores = trimmed_W.std(axis=0)
        snr_scores = torch.abs(mean_scores) / (std_scores + 1e-8)
        return snr_scores
        # return mean_scores

    class WeightScoreLogger:
        def __init__(self, save_dir="weight_scores"):
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
            
            # FOとSNIPのスコアを分けて保存
            self.fo_weight_scores = []
            self.snip_weight_scores = []
            
            # メタデータ
            self.metadata = {
                'timestamp': datetime.now().isoformat(),
                'layers_processed': 0,
                'score_shape': None
            }
        
        def save_fo_layer_scores(self, layer_idx, W_metric):
            """FOの重み単位スコアを保存"""
            # [hidden_dim, intermediate_size] の形状で保存
            self.fo_weight_scores.append({
                'layer_idx': layer_idx,
                'W_metric': W_metric.cpu().clone(),
                'shape': list(W_metric.shape)
            })
            
        def save_snip_layer_scores(self, layer_idx, W_metric):
            """SNIPの重み単位スコアを保存"""
            self.snip_weight_scores.append({
                'layer_idx': layer_idx,
                'W_metric': W_metric.cpu().clone(),
                'shape': list(W_metric.shape)
            })
        
        def save_to_files(self):
            """ファイルに保存"""
            # FOスコア保存
            torch.save(self.fo_weight_scores, 
                    os.path.join(self.save_dir, 'fo_weight_scores.pt'))
            
            # SNIPスコア保存
            torch.save(self.snip_weight_scores, 
                    os.path.join(self.save_dir, 'snip_weight_scores.pt'))
            
            # メタデータ保存
            with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            print(f"Saved weight scores to {self.save_dir}")
            print(f"FO layers: {len(self.fo_weight_scores)}")
            print(f"SNIP layers: {len(self.snip_weight_scores)}")

    global P_SVD_loss 
    P_SVD_loss = torch.zeros(1, requires_grad=True, dtype=torch.float32).to("cpu")
    print("P_SVD_loss shape:", P_SVD_loss.shape)

    logger = WeightScoreLogger(save_dir=f"weight_scores_{args.pruning_ratio}")

    def store_feature(module, input, output):
        global P_SVD_loss
        # if 'LlamaMLP' in module.__class__.__name__:
        if hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif hasattr(output, 'hidden_states'):
            output = output.hidden_states
        elif isinstance(output, tuple):
            output = output[0]

        if output is None:
            return
        output = output.reshape(output.size(0), -1).to(dtype=torch.float32)  # バッチサイズ次元以外をフラットにする
        S = torch.linalg.svdvals(output)  # SVDを計算
        # S = S.to(dtype=torch.float16)  # 特異値をCPUに移動
        singular_value_mean = S.mean().to("cpu",dtype=torch.float32)  # 特異値の平均を計算してP_SVD_lossに加算
        # P_SVD_loss += singular_value_mean
        P_SVD_loss = P_SVD_loss + singular_value_mean

    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(store_feature)
        hooks.append(hook) 

    rm_module = rm_modules(model)
    rm_weights = [module.weight for module, _ in rm_module]  # FOが枝刈りを担当する重み

    with torch.no_grad():
        fo_grads = [torch.zeros_like(w) for w in rm_weights]
        snip_grads = [torch.zeros_like(w) for w in rm_weights]

    for i, (inputs, targets) in enumerate(dataloader):
        # inputs = inputs.cuda(devic  # データをデバイスに転送
        outputs = model(inputs)  # フォワードパス
        break
    
    # for batch in dataloader:
    #     inputs_id = batch["input_ids"]
    #     outputs = model(inputs_id)
    #     break

    # input_batch = torch.stack([item[0] for item in dataloader]).squeeze(1).cuda(device)  # バッチの最初の要素を取得
    # print(f"input_batch shape: {input_batch.shape}")
    # print("start forward pass")
    # outputs = model(input_batch)  # フォワードパス
    # print("forward pass complete")

    print("P_SVD_loss contains nan:", torch.isnan(P_SVD_loss).any().item())
    print("P_SVD_loss contains inf:", torch.isinf(P_SVD_loss).any().item())
    print("P_SVD_loss tensor dtype:", P_SVD_loss.dtype)
    print("P_SVD_loss shape:", P_SVD_loss.shape)
    
    fo_grads = list(torch.autograd.grad(P_SVD_loss, rm_weights))  # FOの勾配を計算
    for i, grad in enumerate(fo_grads):
        has_inf = torch.isinf(grad).any().item()
        has_nan = torch.isnan(grad).any().item()
        if has_inf or has_nan:
            print(f"fo_grads[{i}] contains inf or nan")
        else:
            print(f"fo_grads[{i}] is clean")

    for hook in hooks:
        hook.remove() # メモリ消費えぐいのでhookを外す
    P_SVD_loss = torch.zeros(1)
    model.zero_grad()  # 勾配をリセット
    layers = model.model.layers
    mlp_metric_list = []
    mlp_mask = []
    with torch.no_grad():
        for i in tqdm(range(len(layers)),desc="Processing layers"):
            W_down = rm_weights[i+64] * fo_grads[i+64]#64or80 ブロック数の2倍
            W_up = (rm_weights[i+32] * fo_grads[i+32]).t()#32 or 40 ブロック数と同じ
            W_gate = (rm_weights[i] * fo_grads[i]).t()
            # print(f"W_down: {W_down.shape}, W_up: {W_up.shape}, W_gate: {W_gate.shape}")
            W_metric = W_down + W_up + W_gate
            # print(f"W_metric: {W_metric}")
            # W_metric = W_metric.mean(axis=0)  # 平均を取る
            logger.save_fo_layer_scores(i, W_metric)
            W_metric = torch.abs(W_metric)
            W_metric = calculate_neuron_score_v2(W_metric)
            # print(f"W_metric after calculate_neuron_score_v4: {W_metric.shape}")
            # mlp_metric_list.append(W_metric.cpu())
            mlp_metric_list.append(torch.abs(W_metric).cpu())
            print(f"SVD_W_metric3 {i} have NaN:", torch.isnan(W_metric).any().item())
            print(f"SVD_W_metric3 {i} have inf:", torch.isinf(W_metric).any().item())
    fo_score = torch.stack(mlp_metric_list)
    model.zero_grad()  # 勾配をリセット
    model.eval()  # 評価モードに切り替え
    model.train()
    model.eval()

    for i, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)
        break

    # for batch in dataloader:
    #     inputs_id = batch["input_ids"]
    #     outputs = model(inputs_id)
    #     break
    outputs = outputs.logits
    print("outputs contains nan:", torch.isnan(outputs).any().item())
    print("outputs contains inf:", torch.isinf(outputs).any().item())
    outputs = outputs.reshape(-1, outputs.shape[-1]).to(dtype=torch.float32)
    targets = targets.reshape(-1).to(dtype=torch.long)
    loss = nn.CrossEntropyLoss()(outputs, targets) # CE Loss
    print("before loss contains nan:", torch.isnan(loss).any().item())
    print("before loss contains inf:", torch.isinf(loss).any().item())
    loss = loss.to(dtype=torch.float16)  # 損失をfloat32に変換
    snip_grads = list(torch.autograd.grad(loss, rm_weights)) #SNIPの勾配を計算
    print("after loss contains nan:", torch.isnan(loss).any().item())
    print("after loss contains inf:", torch.isinf(loss).any().item())
    for i, grad in enumerate(snip_grads):
        has_inf = torch.isinf(grad).any().item()
        has_nan = torch.isnan(grad).any().item()
        if has_inf or has_nan:
            print(f"snip_grads[{i}] contains inf or nan")
        else:
            print(f"snip_grads[{i}] is clean")
    mlp_metric_list = []
    mlp_mask = []
    with torch.no_grad():
        for i in tqdm(range(len(layers)),desc="Processing layers"):
            W_down = rm_weights[i+64] * snip_grads[i+64]
            W_up = (rm_weights[i+32] * snip_grads[i+32]).t()
            W_gate = (rm_weights[i] * snip_grads[i]).t()
            W_metric = W_down + W_up + W_gate
            W_metric = torch.abs(W_metric)
            # W_metric = W_metric.mean(axis=0)
            print(f"SNIP_W_metric {i} have NaN:", torch.isnan(W_metric).any().item())
            print(f"SNIP_W_metric {i} have inf:", torch.isinf(W_metric).any().item())
            logger.save_snip_layer_scores(i, W_metric)
            W_metric = calculate_neuron_score_v2(W_metric)
            # mlp_metric_list.append(W_metric.cpu())
            mlp_metric_list.append(torch.abs(W_metric).cpu())
    snip_score = torch.stack(mlp_metric_list)
    print("fo_score: ", fo_score.shape)
    print("fo_score: ", fo_score)
    print("snip_score: ", snip_score.shape)
    print("snip_score: ", snip_score)
    print("fo_score contains nan:", torch.isnan(fo_score).any().item())
    print("fo_score contains inf:", torch.isinf(fo_score).any().item())

    print("snip_score contains nan:", torch.isnan(snip_score).any().item())
    print("snip_score contains inf:", torch.isinf(snip_score).any().item())

    fo_score_standardized = (fo_score - fo_score.mean()) / fo_score.std()
    snip_score_standardized = (snip_score - snip_score.mean()) / snip_score.std()
    print("fo_score_standardized: ", fo_score_standardized.shape)
    print("fo_score_standardized: ", fo_score_standardized)
    print("snip_score_standardized: ", snip_score_standardized.shape)
    print("snip_score_standardized: ", snip_score_standardized)
    
    score = fo_score_standardized + snip_score_standardized
    print("score: ", score.shape)
    print("score: ", score)
    # score_means =  []
    # for i in range(len(layers)):
    #     print(f"score[{i}]: ", score[i].shape)
    #     score_mean = calculate_neuron_score_v2(score[i]) #ここを変えて集約方法変更
    #     score_means.append(score_mean)
    # score = torch.stack(score_means)
    # print("score after mean: ", score.shape)
    
    # sorted_mlp_metric, _ = torch.sort(score.view(-1), descending=True)
    sorted_mlp_metric, _ = torch.sort(score, descending=True)
    thresholds = torch.tensor([sorted_mlp_metric[i][int(sorted_mlp_metric.shape[1]*(1-args.pruning_ratio))] for i in range(len(layers))])
    # limit = sorted_mlp_metric[int(sorted_mlp_metric.shape[0]*(1-args.pruning_ratio))]
    # thresholds = torch.tensor([limit for i in range(len(layers))])
    mlp_mask = (score.t() >= thresholds).t()
    print('*'*30)
    for idx in range(len(layers)):
        compress(model.model.layers[idx], mlp_mask[idx], device)
    
    model.zero_grad()  # 勾配をリセット
    del P_SVD_loss  # グローバル変数をリセット
    P_SVD_loss = torch.zeros(1)  # グローバル変数の再初期化
    logger.save_to_files()
    