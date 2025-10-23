def structed_snip(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    device = [i + 1 for i in range(device - 1)]
    
    print("loading calibdation data")
    dataloader, _ = get_loaders(nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")

    for i, (inp, tar) in enumerate(dataloader):
        outputs = model(inp)
        outputs = outputs.logits
        outputs = outputs.reshape(-1, outputs.shape[-1])
        tar = tar.reshape(-1)
        loss = nn.CrossEntropyLoss()(outputs, tar)
        # grads = list(torch.autograd.grad(loss, rm_weights))
        break

    layers = model.model.layers

    mlp_metric_list = []
    mlp_mask = []

    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
    
        W_down = [(layer.mlp.gate_proj,'weight')]
        W_down = [module.weight for module, _ in W_down]
        W_down_grads = list(torch.autograd.grad(loss, W_down))
        W_up = [(layer.mlp.up_proj,'weight')]
        W_up = [module.weight for module, _ in W_up]
        W_up_grads = list(torch.autograd.grad(loss, W_up))
        W_gate = [(layer.mlp.down_proj,'weight')]
        W_gate = [module.weight for module, _ in W_gate]
        W_gate_grads = list(torch.autograd.grad(loss, W_gate))

        
        W_down_socre = [(weight.cpu() * grad.cpu()).view(-1).abs() for weight, grad in zip(W_down, W_down_grads)]
        W_up_socre = [(weight.cpu() * grad.cpu()).view(-1).abs() for weight, grad in zip(W_up, W_up_grads)]
        W_gate_socre = [(weight.cpu() * grad.cpu()).view(-1).abs() for weight, grad in zip(W_gate, W_gate_grads)]
        W_down_socre = torch.cat(W_down_socre)
        W_up_socre = torch.cat(W_up_socre)
        W_up_socre = W_up_socre.t()
        W_gate_socre = torch.cat(W_gate_socre)
        W_gate_socre = W_gate_socre.t()
        W_metric = W_down_socre + W_up_socre + W_gate_socre
        W_metric = W_metric.mean(axis=0)
        mlp_metric_list.append(W_metric.cpu())

    mlp_metric = torch.stack(mlp_metric_list)

    sorted_mlp_metric, _ = torch.sort(mlp_metric, descending=True)
    # print(sorted_mlp_metric.shape)

    thresholds = torch.tensor([sorted_mlp_metric[i][int(sorted_mlp_metric.shape[1]*args.pruning_ration)] for i in range(len(layers))])
    print(f"thresholds: {thresholds}")


    mlp_mask = (mlp_metric.t() >= thresholds).t()
    print(mlp_mask.shape)

    print('*'*30)
    for idx in range(len(layers)):
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            compress(model.model.layers[idx], mlp_mask[idx], model.hf_device_map[f"model.layers.{idx}"])
        else:
            compress(model.model.layers[idx], mlp_mask[idx], device)

    print('*'*30)
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()