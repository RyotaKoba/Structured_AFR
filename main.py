import sys
import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from importlib.metadata import version

from lib.prune import prune_cfsp, ReFer_L1,ReFer_SVD,check_sparsity,check_sparsity_refer, snip, AFR, structured_snip, Structured_ReFer_SVD, Structured_ReFer_L1, Structured_AFR
from lib.eval import eval_ppl, show_model_input_output
import lib.prune as pruner
from lib.model import rm_modules, all_rm_modules
import torch.nn.utils.prune as prunee
from transformers.models.llama.modeling_llama import LlamaMLP


print('torch', version('torch'))  # 2.1.0
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm_gpu(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        # torch_dtype=torch.float32,
        dtype=torch.float32,
        # trust_remote_code=True,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    if "70B" in args.model or "70b" in args.model: # for 70b we use device_map to load onto multiple GPUs, thus the processing here.
        print(f"args.model: {args.model}")
        device = model.hf_device_map["lm_head"]
        print("use device ", device)
    else:
        device = torch.device("cuda:0")
    model.seqlen = 1024
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    return model, tokenizer, device

def get_llm_cpu(args):
    print("Loading model on CPU")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    print("Model loaded on CPU")
    model.seqlen = 1024
    device = torch.device("cuda:0")
    model.eval()
    print(f"args.model: {args.model}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')    # Huggingface model name
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')

    parser.add_argument('--a', type=float, default=1, help='global control')
    parser.add_argument('--b', type=float, default=1, help='local control')
    parser.add_argument('--c', type=float, default=1, help='local control')
    parser.add_argument('--global_metrics', type=str, default="angular", help='angular, cosine, mse, mae, avg_base')
    parser.add_argument('--local_metrics', type=str, default="three_w_one_wa", help='one_wa, one_a, three_w_one_a, three_w_one_wa, wanda_base, mag_base')

    parser.add_argument('--cuda_friendly', action="store_true")
    parser.add_argument('--pruning_ratio', type=float, default=0, help='Pruning ratio.')
    parser.add_argument("--prune_method", type=str, default="cfsp", choices=["cfsp","refer_l1","refer_svd","snip","structured_snip","structured_refer_svd","structured_refer_l1","structured_afr","afr","none","done"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)

    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--sample', action="store_true")
    parser.add_argument('--all', action="store_true")
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    if torch.cuda.is_available():
        print(" ---- CUDA is available! ------")
    else:
        print(" ---- no cuda! ------")

    # Prune the model
    print("pruning starts")
    
    if args.cuda:
        model, tokenizer, device = get_llm_gpu(args)
    else:
        model, tokenizer, device = get_llm_cpu(args)
    
    print(f"loading llm model {args.model}, pruning method: {args.prune_method}")
    
    if args.prune_method == "cfsp":
        prune_cfsp(args, model, tokenizer, device)#(1,788,2427)
    elif args.prune_method == "structured_snip":
        structured_snip(args, model, tokenizer, device)
    elif args.prune_method == "structured_refer_svd":
        Structured_ReFer_SVD(args, model, tokenizer, device)
    elif args.prune_method == "structured_refer_l1":
        Structured_ReFer_L1(args, model, tokenizer, device)
    elif args.prune_method == "structured_afr":
        Structured_AFR(args, model, tokenizer, device)
    elif args.prune_method == "refer_l1":
        init_data = model.state_dict()
        device = torch.cuda.device_count()
        pruner.SCORE = ReFer_L1(args, model, tokenizer,device)
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,cache_dir=args.cache_dir,device_map=None)
        model.load_state_dict(init_data)
        model.seqlen = 1024
        if args.all:
            rm_module = all_rm_modules(model)
        else:
            rm_module = rm_modules(model)
        pruner.SCORE = pruner.SCORE.float()
        pruner.prune.global_unstructured(rm_module, pruning_method=pruner.Pruner, amount=args.pruning_ratio)
    elif args.prune_method == "refer_svd":
        init_data = model.state_dict()
        device = torch.cuda.device_count()
        pruner.SCORE = ReFer_SVD(args, model, tokenizer,device)
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,cache_dir=args.cache_dir,device_map=None)
        model.load_state_dict(init_data)
        model.seqlen = 1024
        if args.all:
            rm_module = all_rm_modules(model)
        else:
            rm_module = rm_modules(model)
        pruner.SCORE = pruner.SCORE.float()
        pruner.prune.global_unstructured(rm_module, pruning_method=pruner.Pruner, amount=args.pruning_ratio)
    elif args.prune_method == "snip":
        init_data = model.state_dict()
        device = torch.cuda.device_count()
        pruner.SCORE = snip(args, model, tokenizer, device)
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,cache_dir=args.cache_dir,device_map=None)
        model.load_state_dict(init_data)
        model.seqlen = 1024
        if args.all:
            rm_module = all_rm_modules(model)
        else:
            rm_module = rm_modules(model)
        pruner.SCORE = pruner.SCORE.float()
        pruner.prune.global_unstructured(rm_module, pruning_method=pruner.Pruner, amount=args.pruning_ratio)
    elif args.prune_method == "afr":
        init_data = model.state_dict()
        device = torch.cuda.device_count()
        pruner.SCORE = AFR(args, model, tokenizer, device)
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,cache_dir=args.cache_dir,device_map=None)
        model.load_state_dict(init_data)
        model.seqlen = 1024
        if args.all:
            rm_module = all_rm_modules(model)
        else:
            rm_module = rm_modules(model)
        pruner.SCORE = pruner.SCORE.float()
        pruner.prune.global_unstructured(rm_module, pruning_method=pruner.Pruner, amount=args.pruning_ratio)
    elif args.prune_method == "none":
        print(f"loading llm model {args.model} without pruning")
        model.eval()
    elif args.prune_method == "done":
        print(f"loading llm model {args.model} with pruned model")
        model_path = "./pruned_model/Llama3-8B_AFR-St_0.5p_onlyFFN_sizeChanged"
        # model_path = "./llm_weights/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2" #vicuna-model
        # model_path = "./llm_weights/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920" #llama-model
        print("evaluate :", model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map="auto")
        model.seqlen = 1024
        device = torch.device("cuda:0")
        # device = torch.device("cpu")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    print("*"*30)
    # if args.prune_method == "refer_l1" or args.prune_method == "snip" or args.prune_method == "afr" or args.prune_method == "refer_svd" or args.prune_method == "done":
    sparsity_ratio, pruned_model_param = check_sparsity_refer(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print(f"model parameter {pruned_model_param}B")
    print("*"*30)
    
    if args.eval:
        print("Start evaluation")
        ppl = eval_ppl(args, model, tokenizer, device)
        print(f"ppl on wikitext {ppl}")
    
    if args.sample:
        print("Start show sample")
        show_model_input_output(model, tokenizer, device)

    if args.save_model and args.prune_method != "none" and args.prune_method != "done":
        if args.prune_method != "cfsp" and args.prune_method != "structured_snip" and args.prune_method != "structured_refer_svd" and args.prune_method != "structured_refer_l1" and args.prune_method != "structured_afr":
            for module in model.modules():
                if isinstance(module, LlamaMLP):
                    prunee.remove(module.gate_proj, 'weight') 
                    prunee.remove(module.up_proj, 'weight') 
                    prunee.remove(module.down_proj, 'weight') 
            if not os.path.exists(args.save_model):
                os.makedirs(args.save_model)

        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()
