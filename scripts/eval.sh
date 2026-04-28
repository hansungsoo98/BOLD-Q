#!/bin/bash
# Remove previously cached per-layer quantized weights so a new run
# regenerates them from the current model/configuration. Since this
# workflow saves quantized weights in advance and reuses them during
# inference, clearing old layer files helps avoid stale weights from
# earlier experiments.
find ./quantized_layer_weights -type f -name "*layer*" -delete    

CUDA_VISIBLE_DEVICES=0 \
python mx_main.py \
    --model   meta-llama/Llama-2-7b-hf \
    --output_dir ./log \
    --mx \
    --scale_bits 8 \
    --w_elem_format lns4_e2m1 \
    --a_elem_format lns4_e2m1 \
    --block_size 32 \
    --bfloat 16      \
    --lns_mode      \
    --prequantized  \
    --eval_ppl    \
    --smoothquant
