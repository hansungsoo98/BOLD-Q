import torch

from datautils import get_loaders


def act_scaling_hook(module, inputs):
    x = inputs[0]
    scale = module.activation_scale
    for _ in range(x.ndim - 1):
        scale = scale.unsqueeze(0)
    return (x / scale,)


def run_smoothquant(lm, args, logger):
    logger.info("Running SmoothQuant")

    _, calibloader = get_loaders("wikitext2", seed=args.seed, model=args.model, seqlen=lm.seqlen)
    if hasattr(calibloader, "input_ids"):
        calib_inputs = calibloader.input_ids[:, : 32 * lm.seqlen]
    elif isinstance(calibloader, torch.Tensor):
        calib_inputs = calibloader[:, : 32 * lm.seqlen]
    else:
        raise TypeError(f"Unsupported calibloader type: {type(calibloader)}")

    activation_max = {}
    weight_max = {}
    for name, module in lm.model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            activation_max[name] = torch.zeros(module.weight.shape[1], device=lm.device)
            weight_max[name] = module.weight.abs().max(dim=0)[0].detach().cpu()

    with torch.no_grad():
        hooks = []

        def save_acts(name):
            def hook(module, inp, _out):
                if name not in activation_max:
                    return
                acts = inp[0]
                if acts.dim() > 2:
                    acts = acts.reshape(-1, acts.shape[-1])
                activation_max[name] = torch.maximum(
                    activation_max[name],
                    acts.abs().max(dim=0)[0].detach(),
                )

            return hook

        for name, module in lm.model.named_modules():
            if isinstance(module, torch.nn.Linear) and name in activation_max:
                hooks.append(module.register_forward_hook(save_acts(name)))

        for i in range(32):
            batch = calib_inputs[:, (i * lm.seqlen):((i + 1) * lm.seqlen)].to(lm.device)
            lm.model(input_ids=batch)

        for hook in hooks:
            hook.remove()

    alpha = 0.5
    weight_smooth_scale = {}
    act_smooth_scale = {}
    for name in activation_max:
        w_max = weight_max[name]
        a_max = activation_max[name].cpu()
        scale = a_max.pow(alpha) / (w_max.pow(1 - alpha) + 1e-6)
        scale = torch.clamp(scale, min=1e-3, max=1e3)
        if torch.isnan(scale).any() or torch.isinf(scale).any():
            scale = torch.ones_like(scale)
        weight_smooth_scale[name] = scale
        act_smooth_scale[name] = scale

    for name, module in lm.model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in weight_smooth_scale and "lm_head" not in name:
            scale = weight_smooth_scale[name].to(module.weight.device).view(1, -1)
            module.weight.data.mul_(scale)

    for name, module in lm.model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in act_smooth_scale and "lm_head" not in name:
            scale = act_smooth_scale[name].to(module.weight.device)
            module.register_buffer("activation_scale", scale)
            module.register_forward_pre_hook(act_scaling_hook)

    logger.info("SmoothQuant finished")
