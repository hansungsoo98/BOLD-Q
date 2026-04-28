import copy
import gc
from contextlib import nullcontext

import torch
import torch.nn as nn

from models.int_falcon_layer import QuantFalconDecoderLayer
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quant.int_linear import QuantLinear


def add_new_module(name, original_module, added_module):
    """Recursively locate a module path and replace it with a new module."""
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)


def vanilla_quant(lm, args, logger=None, calibration_data=None):
    """
    Basic vanilla quantization routine.
    Uses calibration data to collect activation statistics before quantization.
    """
    logger.info("Starting vanilla quantization...")

    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Select the layer stack based on model type.
    if "llama" in args.net.lower():
        layers = model.model.layers
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        layer_name_prefix = "model.transformer.h"
    else:
        raise ValueError("Unsupported model type. Supported types: llama, opt, falcon.")

    # Step 1: collect activation statistics with calibration data.
    logger.info("Running calibration...")
    for batch in calibration_data:
        with torch.no_grad():
            _ = model(batch.to(dev))
    logger.info("Calibration complete.")

    for i in range(len(layers)):
        logger.info(f"=== Quantizing layer {i} ===")
        layer = layers[i].to(dev)

        # Apply quantized linear layers and activation quantizers.
        qlayer = copy.deepcopy(layer)
        for name, module in qlayer.named_modules():
            if isinstance(module, torch.nn.Linear):
                quant_linear = QuantLinear(
                    module, args.weight_quant_params, args.act_quant_params
                )
                add_new_module(name, qlayer, quant_linear)
            elif isinstance(module, nn.ReLU) or isinstance(module, nn.GELU):
                quant_act = QuantAct(act_bit=args.abits)
                add_new_module(name, qlayer, quant_act)

        layers[i] = qlayer.to("cpu")
        del layer
        torch.cuda.empty_cache()

    gc.collect()
    model.config.use_cache = use_cache
    return model
