import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from safetensors.torch import load_file, save_file
from tqdm import tqdm

import utils
from datautils import get_loaders
from models.LMClass import LMClass
from quant.mx import finalize_mx_specs, mx_mapping
from quant.mx.elemwise_ops import quantize_elemwise_op
from quant.mx.mx_ops import quantize_mx_op
from quant.smoothquant import run_smoothquant

torch.backends.cudnn.benchmark = True

WEIGHT_SAVE_DIR = "quantized_layer_weights"
os.makedirs(WEIGHT_SAVE_DIR, exist_ok=True)


def infer_model_backend(model_name: str) -> str:
    lower_name = model_name.lower()
    if "opt" in lower_name:
        return "opt"
    if "falcon" in lower_name:
        return "falcon"
    return "llama"


def get_hidden_states(lm, backend: str, batch: torch.Tensor):
    if backend == "opt":
        outputs = lm.model.model.decoder(batch)
    elif backend == "falcon":
        outputs = lm.model.transformer(batch)
    else:
        outputs = lm.model.model(batch)
    return outputs[0]


def load_or_create_testloader(lm, args, logger, dataset: str):
    sanitized_model_name = args.model.replace("/", "_")
    cache_path = Path(args.cache_dir) / f"testloader_{sanitized_model_name}_{dataset}.cache"

    if cache_path.exists():
        try:
            testloader = torch.load(cache_path)
            logger.info(f"Loaded cached test data from {cache_path}")
            return testloader
        except Exception as exc:
            logger.warning(f"Failed to load cache {cache_path}: {exc}")
            try:
                cache_path.unlink()
            except OSError:
                pass

    logger.info(f"Generating test data for {dataset}")
    _, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=lm.seqlen)

    try:
        torch.save(testloader, cache_path)
        logger.info(f"Saved test data to {cache_path}")
    except Exception as exc:
        logger.warning(f"Failed to save cache {cache_path}: {exc}")

    return testloader


@torch.no_grad()
def evaluate_perplexity(lm, args, logger):
    dataset = "wikitext2"
    backend = infer_model_backend(args.model)
    testloader = load_or_create_testloader(lm, args, logger, dataset)

    if isinstance(testloader, torch.Tensor):
        testenc = testloader
    elif hasattr(testloader, "input_ids"):
        testenc = testloader.input_ids
    else:
        raise TypeError(f"Unsupported testloader type: {type(testloader)}")

    nsamples = testenc.numel() // lm.seqlen
    use_cache = lm.model.config.use_cache
    lm.model.config.use_cache = False
    lm.model.eval()

    nlls = []
    for i in tqdm(range(nsamples), desc=f"Evaluating {dataset}"):
        batch = testenc[:, (i * lm.seqlen):((i + 1) * lm.seqlen)].to(lm.device)
        hidden_states = get_hidden_states(lm, backend, batch)
        logits = lm.model.lm_head(hidden_states)
        shift_logits = logits[:, :-1, :]
        shift_labels = testenc[:, (i * lm.seqlen):((i + 1) * lm.seqlen)][:, 1:].to(
            lm.model.lm_head.weight.device
        )
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * lm.seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
    lm.model.config.use_cache = use_cache
    logger.info(f"{dataset}: {ppl.item()}")
    return {dataset: ppl.item()}


def quantize_and_save_layer_weight(layer_name, module, mx_specs, device):
    if "lm_head" in layer_name:
        return

    with torch.no_grad():
        module.to(device)
        weight = module.weight.to(device)
        quantized_weight = quantize_elemwise_op(
            weight,
            mx_specs=mx_specs,
            round=mx_specs["round_weight"],
        )
        quantized_weight = quantize_mx_op(
            quantized_weight,
            mx_specs,
            elem_format=mx_specs["w_elem_format"],
            axes=[-1],
            round=mx_specs["round_mx_output"],
        )

        save_path = os.path.join(
            WEIGHT_SAVE_DIR,
            f"{layer_name.replace('.', '_')}.safetensors",
        )
        save_file({"weight": quantized_weight.detach().cpu()}, save_path)


def process_layer_queue(layer_list, mx_specs, device):
    device_count = torch.cuda.device_count()
    if isinstance(device, int):
        if device < 0 or device >= device_count:
            raise ValueError(f"Invalid CUDA device index {device}. Available: 0 to {device_count - 1}")
        torch.cuda.set_device(device)
        device = torch.device(f"cuda:{device}")

    for layer_name, module in layer_list:
        quantize_and_save_layer_weight(layer_name, module, mx_specs, device)


def quantize_and_save_weights_parallel(model, mx_specs, devices):
    available_gpus = []
    for device in devices:
        if isinstance(device, str) and device.startswith("cuda:"):
            index = int(device.split(":")[1])
        else:
            index = int(device)
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid CUDA device index {index}. Available: 0 to {torch.cuda.device_count() - 1}"
            )
        available_gpus.append(index)

    layer_lists = [[] for _ in available_gpus]
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            layer_lists[i % len(available_gpus)].append((name, module))

    processes = []
    for i, gpu_index in enumerate(available_gpus):
        process = mp.Process(target=process_layer_queue, args=(layer_lists[i], mx_specs, gpu_index))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


def load_and_replace_layer_weights(model, load_dir=WEIGHT_SAVE_DIR):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            load_path = os.path.join(load_dir, f"{name.replace('.', '_')}.safetensors")
            if os.path.exists(load_path):
                with torch.no_grad():
                    data = load_file(load_path)
                    module.weight.data.copy_(data["weight"].to(module.weight.device))
    return model


def patch_layernorm_fp32(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            module.float()
            original_forward = module.forward

            def forward_fp32(x, *args, _module=module, _original=original_forward, **kwargs):
                output = _original(x.to(_module.weight.dtype), *args, **kwargs)
                return output.to(x.dtype)

            module.forward = forward_fp32


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./log", help="Directory for logs")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory for cached evaluation data")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--seed", type=int, default=2, help="Random seed")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation used when loading the model",
    )
    parser.add_argument("--eval_ppl", action="store_true", help="Run perplexity evaluation")
    parser.add_argument("--mx", action="store_true", help="Enable MX quantized ops")
    parser.add_argument("--prequantized", action="store_true", help="Pre-quantize and reload linear weights")
    parser.add_argument("--smoothquant", action="store_true", help="Apply SmoothQuant before MX injection")
    parser.add_argument("--scale_bits", type=int, default=8, help="Shared exponent bit width")
    parser.add_argument("--w_elem_format", type=str, default="int8", help="Weight MX element format")
    parser.add_argument("--a_elem_format", type=str, default="int8", help="Activation MX element format")
    parser.add_argument("--block_size", type=int, default=16, help="MX block size")
    parser.add_argument("--bfloat", type=int, default=16, help="BFloat mantissa format")
    parser.add_argument("--lns_mode", action="store_true", help="Enable LNS mode")
    return parser

def main():
    mp.set_start_method("spawn", force=True)
    args = build_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    logger = utils.create_logger(Path(args.output_dir))
    logger.info(args)

    lm = LMClass(args)
    lm.seqlen = 2048
    lm.model = lm.model.to(lm.device)
    patch_layernorm_fp32(lm.model)
    lm.model.eval()

    for param in lm.model.parameters():
        param.requires_grad = False

    if args.smoothquant:
        run_smoothquant(lm, args, logger)

    mx_specs = finalize_mx_specs(
        {
            "scale_bits": args.scale_bits,
            "w_elem_format": args.w_elem_format,
            "a_elem_format": args.a_elem_format,
            "block_size": args.block_size,
            "bfloat": args.bfloat,
            "lns_mode": args.lns_mode,
            "prequantized": args.prequantized,
        }
    )

    if args.prequantized:
        quantize_and_save_weights_parallel(lm.model, mx_specs, ["cuda:0"])
        lm.model = load_and_replace_layer_weights(lm.model)

    if args.mx:
        logger.info("Injecting MX quantized ops")
        tick = time.time()
        mx_mapping.inject_pyt_ops(mx_specs)
        logger.info(f"MX op injection finished in {time.time() - tick:.2f}s")

    if args.eval_ppl:
        evaluate_perplexity(lm, args, logger)


if __name__ == "__main__":
    print(sys.argv)
    main()
