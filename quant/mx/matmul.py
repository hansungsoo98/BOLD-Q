"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch
import numpy as np

from .mx_ops import quantize_mx_op, quantize_mx_op_matmul_A, quantize_mx_op_matmul_B
from .elemwise_ops import quantize_elemwise_op
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test
from .matmul_precision import set_matmul_precision
from typing import Union

torch_matmul = torch.matmul
torch_addmm = torch.addmm


def inject_blockwise_mul_noise_torch(out: "torch.Tensor",
                                        block_size: int = 32,
                                        rel_mu: float = 0.0,
                                        rel_sigma: float = 1e-4,
                                        clamp_std: Union[float, None] = 4.0,
                                        generator: "torch.Generator | None" = None):
    """Injects block-wise multiplicative noise into a PyTorch tensor."""
    *lead, C = out.shape
    device, dtype = out.device, out.dtype
    n_blocks = (C + block_size - 1) // block_size
    
    if n_blocks == 0:
        return out
        
    z = torch.randn(*lead, n_blocks, device=device, dtype=dtype, generator=generator)
    noise_blocks = 1.0 + (rel_mu + rel_sigma * z)
    
    if clamp_std is not None and clamp_std > 0:
        lo = 1.0 + rel_mu - clamp_std * rel_sigma
        hi = 1.0 + rel_mu + clamp_std * rel_sigma
        noise_blocks = noise_blocks.clamp(min=lo, max=hi)
        
    noise_full = noise_blocks.repeat_interleave(block_size, dim=-1)[..., :C]
    return out * noise_full


class MatMulFunction(torch.autograd.Function):
    """Matches functionality of torch.matmul. Attempts to broadcast
    outmost dims if in1 and in2 have the same number of dims.
        in1: (..., out_rows, features)
        in2: (..., features, out_cols)
        out: (..., out_rows, out_cols)
    Otherwise, it expects the following shapes:
        in1: (..., out_rows, features)
        in2: (features, out_cols)
        out: (..., out_rows, out_cols)
    """

    @staticmethod
    def forward(ctx, in1, in2, bias, mx_specs, name, mode_config='aa'):
        assert mode_config in ["aa", "aw", "wa"]
        ctx.mode_config = mode_config
        if mode_config[0] == "a":
            qin1_elem_format = mx_specs["a_elem_format"]
        else:
            qin1_elem_format = mx_specs["w_elem_format"]

        if mode_config[1] == "a":
            qin2_elem_format = mx_specs["a_elem_format"]
        else:
            qin2_elem_format = mx_specs["w_elem_format"]

        bf_in1 = quantize_elemwise_op(
            in1, mx_specs=mx_specs, round=mx_specs["round_output"]
        )
        bf_in2 = quantize_elemwise_op(
            in2, mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        if bias is not None:
            bf_bias = quantize_elemwise_op(
                bias, mx_specs=mx_specs, round=mx_specs["round_weight"]
            )

            ctx.bias_shape = list(bias.shape)
        else:
            bf_bias = None
            ctx.bias_shape = None

        if mx_specs["quantize_backprop"]:
            ctx.save_for_backward(bf_in1, bf_in2)
        else:
            ctx.save_for_backward(in1, in2)

        # quantize along the dot product dimension
        # print(qin1_elem_format)
        qin1 = quantize_mx_op_matmul_A(
            bf_in1,
            mx_specs,
            elem_format=qin1_elem_format,
            axes=[-1],
            round=mx_specs["round_mx_output"],
        )
        # print(qin2_elem_format)
        qin2 = quantize_mx_op_matmul_B(
            bf_in2,
            mx_specs,
            elem_format=qin2_elem_format,
            axes=[-2],
            round=mx_specs["round_mx_output"],
        )

        with set_matmul_precision(qin1, qin2,
                        qin1_elem_format,
                        qin2_elem_format):

            out = torch_matmul(qin1, qin2)

        # noise injection
        out = inject_blockwise_mul_noise_torch(
                out=out,
                block_size=mx_specs.get("noise_block_size", 32),
                rel_mu=mx_specs.get("noise_mu", 4.06e-5),
                rel_sigma=mx_specs.get("noise_sigma", 4.45e-3),
                clamp_std=4.0
        )
        
        out = quantize_elemwise_op(
            out, mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        if bias is not None:
            out = out + bf_bias
            out = quantize_elemwise_op(
                out, mx_specs=mx_specs, round=mx_specs["round_output"]
            )

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        return out

    @staticmethod
    def backward(ctx, grad_out):

        """
        For a matmul in "wa" mode, the fwd and bwd matmuls configs are:
            FWD wt x act: w x a
            BWD wt x grad: w x a
            BWD act x grad: a x a <-- no mixed precision!
        """
        if ctx.mode_config[0] == "a":
            qin1_elem_format = ctx.mx_specs["a_elem_format_bp"]
        else:
            qin1_elem_format = ctx.mx_specs["w_elem_format_bp"]

        if ctx.mode_config[1] == "a":
            qin2_elem_format = ctx.mx_specs["a_elem_format_bp"]
        else:
            qin2_elem_format = ctx.mx_specs["w_elem_format_bp"]

        in1, in2 = ctx.saved_tensors

        grad_out = quantize_elemwise_op(
            grad_out,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # perform madtile operation for grad_in1, grad_in2
        #####################################################
        qin1 = quantize_mx_op(
            in1,
            ctx.mx_specs,
            elem_format=qin1_elem_format,
            axes=[-2],
            round=ctx.mx_specs["round_mx_input_grad_input"],
        )
        qin2 = quantize_mx_op(
            in2,
            ctx.mx_specs,
            elem_format=qin2_elem_format,
            axes=[-1],
            round=ctx.mx_specs["round_mx_input_grad_input"],
        )

        # quantize along out_cols
        qgrad_out1 = quantize_mx_op(
            grad_out,
            ctx.mx_specs,
            elem_format=ctx.mx_specs["a_elem_format_bp_os"],
            axes=[-1],
            round=ctx.mx_specs["round_mx_grad_output_grad_input"],
        )
        # quantize along out_rows
        qgrad_out2 = quantize_mx_op(
            grad_out,
            ctx.mx_specs,
            elem_format=ctx.mx_specs["a_elem_format_bp_os"],
            axes=[-2],
            round=ctx.mx_specs["round_mx_grad_output_grad_input"],
        )

        # compute grad_in1 and grad_in2
        with set_matmul_precision(qgrad_out1, qin2,
            ctx.mx_specs["a_elem_format_bp_os"],
            qin2_elem_format):
            grad_in1 = torch_matmul(qgrad_out1, qin2.transpose(-1, -2))
        
        with set_matmul_precision(qin1, qgrad_out2,
            qin1_elem_format,
            ctx.mx_specs["a_elem_format_bp_os"]):
            grad_in2 = torch_matmul(qin1.transpose(-1, -2), qgrad_out2)

        # element-wise quantize for grad_in1 and grad_in2
        grad_in1 = quantize_elemwise_op(
            grad_in1,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )
        grad_in2 = quantize_elemwise_op(
            grad_in2,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # Compute grad_bias
        #####################################################
        if ctx.bias_shape is None:
            grad_bias = None
        else:
            inner_size = grad_out.shape[-1]
            assert np.prod(ctx.bias_shape) == inner_size
            grad_bias = grad_out.reshape(-1, inner_size).sum(0)
            grad_bias = grad_bias.reshape(ctx.bias_shape)

            grad_bias = quantize_elemwise_op(
                grad_bias,
                mx_specs=ctx.mx_specs,
                round=ctx.mx_specs["round_grad_weight"],
            )

        return (grad_in1, grad_in2, grad_bias, None, None, None)


def matmul(in1, in2, bias=None, mx_specs=None, name=None, mode_config='aw'):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        if bias is None:
            out = torch_matmul(in1, in2)
        else:
            out = torch_addmm(bias, in1, in2)
        return out

    mx_specs = apply_mx_specs(mx_specs)

    return MatMulFunction.apply(in1, in2, bias, mx_specs, name, mode_config)
