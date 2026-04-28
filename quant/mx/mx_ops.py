"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Name:    mx_ops.py

Pytorch methods for MX quantization.

Usage Notes:
 - Use the "Exposed Methods" below to implement autograd functions
 - Use autograd functions to then implement torch.nn.Module(s)
 - Do *not* use methods in this file in Modules, they have no defined
   backwards pass and will block gradient computation.
 - Avoid importing internal function if at all possible.

Exposed Methods:
    quantize_mx_op - quantizes a tensor to MX format.

Internal Methods:
    _safe_lshift, _safe_rshift - fp16 compatible shifts
    _shared_exponents - Returns MX shared exponent for the passed tensor
    _reshape_to_blocks - tiles a tensor by splitting one dim into two
    _undo_reshape_to_blocks - undos the above reshaping
    _quantize_mx - quantizes a tensor to MX format
"""

import os
import torch
import numpy as np
import math
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict ,  Any
from enum import Enum # Assuming ElemFormat is an Enum

from .specs import mx_assert_test
from .formats import (
        RoundingMode,
        ElemFormat,
        FP32_EXPONENT_BIAS,
        FP32_MIN_NORMAL,
        _get_format_params
)
from .elemwise_ops import (
        _safe_lshift, _safe_rshift,
        _round_mantissa,
        _quantize_elemwise_core,
        _quantize_elemwise_lns_core,
        quantize_lns_with_lut,
        _quantize_elemwise_lns_core_sf_max_mapping,
)
import logging
# logger = logger or logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Helper funcs
# -------------------------------------------------------------------------
def _shared_exponents(A, method="max", axes=None, ebits=0, is_lns=False):
    """
    Get shared exponents for the passed matrix A.
    Args:
      A      {PyTorch tensor} -- Input tensor
      method {str}            -- Exponent selection method.
                                 "max" uses the max absolute value
                                 "none" uses an exponent for each value (i.e., no sharing)
      axes   {list(int)}      -- List of integers which specifies the axes across which
                                 shared exponents are calculated.
    Returns:
      shared_exp {PyTorch tensor} -- Tensor of shared exponents
    """

    if method == "max":
        if axes is None:
            shared_exp = torch.max(torch.abs(A))
        else:
            shared_exp = A
            for axis in axes:
                shared_exp, _ = torch.max(torch.abs(shared_exp), dim=axis, keepdim=True)
    elif method == "none":
        shared_exp = torch.abs(A)
    else:
        raise Exception("Unrecognized shared exponent selection method %s" % (method))

    # log2(shared_exp) and truncate to integer
    shared_exp = torch.floor(
        torch.log2(
            shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
        )
    )

    # Restrict to [-emax, emax] range
    if not is_lns and ebits > 0:
        emax = 2**(ebits-1) - 1
        #shared_exp = torch.clamp(shared_exp, -emax, emax)
        # Overflow to Inf
        shared_exp[shared_exp > emax] = float("NaN")
        # Underflows are set to -127 which causes them to be
        # flushed to 0 later
        shared_exp[shared_exp < -emax] = -emax

    return shared_exp


def _reshape_to_blocks(A, axes, block_size):
    if axes is None:
        raise Exception(
            "axes required in order to determine which "
            "dimension to apply block size to"
        )
    if block_size == 0:
        raise Exception("block_size == 0 in _reshape_to_blocks")

    # Fix axes to be positive and sort them
    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    assert all(x >= 0 for x in axes)
    axes = sorted(axes)

    # Add extra dimension for tiles
    for i in range(len(axes)):
        axes[i] += i  # Shift axes due to added dimensions
        A = torch.unsqueeze(A, dim=axes[i] + 1)

    # Pad to block_size
    orig_shape = A.size()
    pad = []
    for i in range(len(orig_shape)):
        pad += [0, 0]

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.value)
        # Don't pad if the axis is short enough to fit inside one tile
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        A = torch.nn.functional.pad(A, pad, mode="constant")

    def _reshape(shape, reshape_block_size):
        for axis in axes:
            # Reshape to tiles if axis length > reshape_block_size
            if shape[axis] >= reshape_block_size:
                assert shape[axis] % reshape_block_size == 0
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            # Otherwise preserve length and insert a 1 into the shape
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    # Reshape to tiles
    padded_shape = A.size()
    reshape = _reshape(list(padded_shape), block_size)

    A = A.view(reshape)
    return A, axes, orig_shape, padded_shape


def _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes):
    # Undo tile reshaping
    A = A.reshape(padded_shape)
    # Undo padding
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        A = A[slices]
    for axis in reversed(axes):
        # Remove extra dimension
        A = torch.squeeze(A, dim=axis + 1)
    return A

def _generate_base_lns_lut_for_core(
    bits: int,
    exp_bits: int,
    device: torch.device,
) -> torch.Tensor:
    """
    """
    emax = 2 ** (exp_bits - 1) - 1 if exp_bits > 0 else 0
    
    if exp_bits == 0:
        emin = 0
        emax = 0
    else:
        emin = -emax - 1

    if bits == 0:
        mvals = [0.0]
    else:
        mvals = [i / (2 ** bits) for i in range(2 ** bits)]

    lns_vals_set = {0.0}

    if int(emin) <= int(emax):
        for s_sign in [1.0, -1.0]:
            for e_int in range(int(emin), int(emax) + 1):
                for m_frac in mvals:
                    if exp_bits > 0 and e_int == emin and m_frac == 0.0 and bits > 0:
                        continue
                    
                    try:
                        val = s_sign * (2.0 ** (e_int + m_frac))
                        if not math.isfinite(val):
                            continue
                        lns_vals_set.add(val)
                    except OverflowError:
                        continue
    
    if not lns_vals_set:
        final_lut = torch.tensor([0.0], dtype=torch.float32, device=device)
    else:
        final_lut = torch.tensor(sorted(list(lns_vals_set)), dtype=torch.float32, device=device)

    if final_lut.numel() == 0 or (final_lut.numel()==1 and torch.all(final_lut==0.0)):
        # print(f"Warning (_generate_base_lns_lut): Generated LUT for E{exp_bits}M{bits} is empty or all zeros. Returning [0.0].")
        return torch.tensor([0.0], dtype=torch.float32, device=device)
        
    return final_lut
def _generate_biased_lns_lut_diff(
    bits: int,
    exp_bits: int,
    bias: float,
    diff: float,
    device: torch.device,
    verbose: bool = False
) -> torch.Tensor:

    emax = 2 ** (exp_bits - 1) - 1 if exp_bits > 0 else 0
    emin = -emax - 1 if exp_bits > 0 else 0
    
    mvals = [i / (2 ** bits) for i in range(2 ** bits)]
    lns_vals = {0.0}
    skipped_count = 0
    if verbose: print(f"--- LUT Gen for E{exp_bits}M{bits}, Bias={bias:.4f}, Diff={diff:.4f} ---")

    for s in [1.0, -1.0]:
        for e_int in range(int(emin), int(emax) + 1):
            for m_frac in mvals:
                if exp_bits > 0 and e_int == emin and m_frac == 0.0:
                    # if verbose: print(f"    Skipping e_int={e_int}, m_frac={m_frac} (denormal-like for LNS)")
                    continue
                
                
                if e_int >= 1:
                    if m_frac > 0:
                        biased_exponent = float(e_int + m_frac + bias)
                    else:
                        biased_exponent = float(e_int + m_frac + bias - diff) 
                else: 
                    if e_int == -2 and m_frac > 0:
                        biased_exponent = float(e_int)
                    else:
                        biased_exponent = float(e_int + m_frac)


                try:
                    val = s * (2.0 ** biased_exponent)
                    # if verbose: print(f"      val = {val:.4e}")
                    if not math.isfinite(val):
                        skipped_count += 1
                        continue
                    lns_vals.add(val)
                except OverflowError:
                    skipped_count += 1
                    continue
    
    final_lut = torch.tensor(sorted(list(lns_vals)), dtype=torch.float32, device=device)

    
    if skipped_count > 0: print(f"    Skipped {skipped_count} invalid values.")
    if final_lut.numel() == 0: print(f"    Error: No valid LUT values generated!")
        # else: print(f"    LUT Size={final_lut.numel()}. Range=[{final_lut.min():.4f}, {final_lut.max():.4f}]")
    
    return final_lut if final_lut.numel() > 0 else torch.tensor([0.0], dtype=torch.float32, device=device)
def _generate_biased_lns_lut_diff_activation(
    mbits: int,
    ebits: int,
    bias_tensor: torch.Tensor,
    diff_tensor: torch.Tensor,
    device: torch.device,
    verbose: bool = False
) -> torch.Tensor:
    """
    """
    num_m = 2 ** mbits
    num_e = 2 ** (ebits - 1)
    emax = num_e - 1 if ebits > 0 else 0
    emin = -emax - 1 if ebits > 0 else 0
    e_range = torch.arange(emin, emax + 1, device=device, dtype=torch.float32)  # [E]
    m_range = torch.arange(num_m, device=device, dtype=torch.float32) / num_m    # [M]
    s_range = torch.tensor([1.0, -1.0], device=device, dtype=torch.float32)      # [2]

    # 2. bias/diff shape [B, 1, 1, 1]
    B = bias_tensor.shape[0]
    bias = bias_tensor.view(B, 1, 1, 1)
    diff = diff_tensor.view(B, 1, 1, 1)

    # 3. e_int/m_frac shape [1, E, 1, 1], [1, 1, M, 1], s [1, 1, 1, 2]
    e_int = e_range.view(1, -1, 1, 1)
    m_frac = m_range.view(1, 1, -1, 1)
    s = s_range.view(1, 1, 1, 2)

    is_e_min_and_m0 = ((e_int == emin) & (m_frac == 0))
    e_ge_1 = (e_int >= 1)
    m_gt_0 = (m_frac > 0)
    m_eq_0 = (m_frac == 0)
    e_eq_m2 = (e_int == -2)

    bias_exp = torch.where(
        e_ge_1,
        torch.where(
            m_gt_0, e_int + m_frac + bias + diff, e_int + m_frac + diff
        ),
        torch.where(
            e_eq_m2 & m_gt_0, e_int, e_int + m_frac
        )
    )

    bias_exp = torch.where(is_e_min_and_m0, torch.full_like(bias_exp, float('nan')), bias_exp)
    lut_vals = s * torch.pow(2.0, bias_exp)

    lut_vals = lut_vals.masked_fill(~torch.isfinite(lut_vals), 0.0)  # nan, inf => 0

    lut_vals_flat = lut_vals.reshape(B, -1)

    lut_cat = torch.cat([lut_vals_flat, torch.zeros(B, 1, device=device)], dim=1)
    return lut_cat

def _generate_biased_lns_lut_bias(
    bits: int,
    exp_bits: int,
    bias: float,
    diff: float,
    device: torch.device,
    verbose: bool = False
) -> torch.Tensor:

    emax = 2 ** (exp_bits - 1) - 1 if exp_bits > 0 else 0
    emin = -emax - 1 if exp_bits > 0 else 0
    
    mvals = [i / (2 ** bits) for i in range(2 ** bits)]
    lns_vals = {0.0}
    skipped_count = 0
    if verbose: print(f"--- LUT Gen for E{exp_bits}M{bits}, Bias={bias:.4f}, Diff={diff:.4f} ---")

    for s in [1.0, -1.0]:
        for e_int in range(int(emin), int(emax) + 1):
            for m_frac in mvals:
                if exp_bits > 0 and e_int == emin and m_frac == 0.0:
                    # if verbose: print(f"    Skipping e_int={e_int}, m_frac={m_frac} (denormal-like for LNS)")
                    continue
                
                
                if e_int >= 1:
                    if m_frac > 0:
                        biased_exponent = float(e_int + m_frac + bias)
                    else:
                        biased_exponent = float(e_int + m_frac) 
                else: 
                    if e_int == -2 and m_frac > 0:
                        biased_exponent = float(e_int)
                    else:
                        biased_exponent = float(e_int + m_frac)


                try:
                    val = s * (2.0 ** biased_exponent)
                    # if verbose: print(f"      val = {val:.4e}")
                    if not math.isfinite(val):
                        skipped_count += 1
                        continue
                    lns_vals.add(val)
                except OverflowError:
                    skipped_count += 1
                    continue
    
    final_lut = torch.tensor(sorted(list(lns_vals)), dtype=torch.float32, device=device)

    
    if skipped_count > 0: print(f"    Skipped {skipped_count} invalid values.")
    if final_lut.numel() == 0: print(f"    Error: No valid LUT values generated!")
        # else: print(f"    LUT Size={final_lut.numel()}. Range=[{final_lut.min():.4f}, {final_lut.max():.4f}]")
    
    return final_lut if final_lut.numel() > 0 else torch.tensor([0.0], dtype=torch.float32, device=device)


def _generate_biased_lns_lut(bits: int, exp_bits: int, bias: float, device: torch.device, verbose: bool = False) -> torch.Tensor:
    emax = 2 ** (exp_bits - 1) - 1 if exp_bits > 0 else 0
    emin = -emax - 1 if exp_bits > 0 else 0
    mvals = [i / (2 ** bits) for i in range(2 ** bits)]
    lns_vals = {0.0}
    skipped_count = 0

    for s in [1.0, -1.0]:
        for e in range(int(emin), int(emax) + 1):
            for m in mvals:
                if e == emin and m == 0: continue

                if e >= 0:
                    biased_exponent = float(e + m + bias)
                else:
                    biased_exponent = float(e + m )

                try:
                    # val = s * math.exp2(biased_exponent)
                    val = s * (2.0 ** biased_exponent)
                    if not math.isfinite(val):
                        skipped_count += 1
                        continue
                    lns_vals.add(val)
                except OverflowError:
                    skipped_count += 1
                    continue

    final_lut = torch.tensor(sorted(list(lns_vals)), dtype=torch.float32, device=device)

    if verbose:
        if skipped_count > 0: print(f"    [LUT Gen Bias={bias:.4f}] Skipped {skipped_count} invalid values.")
        if final_lut.numel() == 0: print(f"    Error: No valid LUT values generated for bias={bias:.4f}!")
        else: print(f"    Info [LUT Gen Bias={bias:.4f}]: Size={final_lut.numel()}. Range=[{final_lut.min():.4f}, {final_lut.max():.4f}]")

    if final_lut.numel() == 0: return torch.tensor([0.0], dtype=torch.float32, device=device)
    return final_lut

def _snap_to_lut(tensor_in: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    if lut.numel() == 0: return torch.zeros_like(tensor_in)

    tensor_flat = tensor_in.flatten().to(torch.float32)
    snapped = torch.empty_like(tensor_flat)
    chunk_size = 32768 * 4

    for start in range(0, tensor_flat.shape[0], chunk_size):
        end = start + chunk_size
        chunk = tensor_flat[start:end].unsqueeze(1)
        dists = torch.abs(chunk - lut.unsqueeze(0))
        idx = torch.argmin(dists, dim=1)
        snapped[start:end] = lut[idx]

    return snapped.view_as(tensor_in).to(tensor_in.dtype)


def print_quant_param_statistics_v3(
    best_format_indices: torch.Tensor,
    best_bias_indices: torch.Tensor,
    best_diff_indices: torch.Tensor,
    candidate_formats_list: List[tuple],
    bias_values_list: List[float],
    diff_values_list: List[float],
):
    """Print compact debug statistics for selected quantization parameters."""
    total_blocks = best_format_indices.numel()
    if total_blocks == 0:
        print("--- Quantization Parameter Statistics ---")
        print("No blocks to analyze.")
        return

    print("\\n--- Quantization Parameter Statistics ---")
    print(f"Total blocks analyzed: {total_blocks}")

def _expand_scale_like_A(scale: torch.Tensor, target_tensor: torch.Tensor, re_axes: Optional[List[int]]) -> torch.Tensor:
    """Expand a reduced scale tensor so it matches the target tensor shape."""
    if not re_axes:
        if scale.numel() == 1:
            return scale.expand_as(target_tensor)
        if scale.shape == target_tensor.shape:
            return scale
        raise ValueError(
            f"Per-element scale shape {scale.shape} must match target shape {target_tensor.shape}"
        )

    target_ndim = target_tensor.ndim
    re_axes_pos = sorted([ax + target_ndim if ax < 0 else ax for ax in re_axes])
    expected_scale_ndim = target_ndim - len(re_axes_pos)
    if scale.ndim != expected_scale_ndim:
        raise ValueError(
            f"Scale ndim ({scale.ndim}) is inconsistent. "
            f"Expected {expected_scale_ndim} (target_ndim {target_ndim} - num_re_axes {len(re_axes_pos)})."
        )

    scale_expanded = scale
    for ax in re_axes_pos:
        scale_expanded = scale_expanded.unsqueeze(ax)
    return scale_expanded.expand_as(target_tensor)

def _snap_to_lut_activation(data_batch: torch.Tensor, lut_batch: torch.Tensor) -> torch.Tensor:
    """
    (Vectorized Version)

    Args:

    Returns:
    """
    # data_batch: [N, block_size] -> [N, block_size, 1]
    # lut_batch:  [N, lut_size]  -> [N, 1, lut_size]
    data_expanded = data_batch.unsqueeze(-1)
    lut_expanded = lut_batch.unsqueeze(-2)

    dists = torch.abs(data_expanded - lut_expanded)

    indices = torch.argmin(dists, dim=-1)  # shape: [N, block_size]

    snapped_values = torch.gather(lut_batch, -1, indices)

    return snapped_values


# -------------------------------------------------------------------------
# Main funcs
# -------------------------------------------------------------------------
def _quantize_mx(
    A,
    scale_bits,
    elem_format,    # can be None for no quantization
    shared_exp_method="max",
    axes=None,
    block_size=0,
    round="nearest",
    flush_fp32_subnorms=False,
    custom_cuda=False,
    lns_mode=False,
    verbose_lut: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Quantize a tensor in MX format."""

    best_bias_index_map_out: Optional[torch.Tensor] = None

    if elem_format is None: return A, best_bias_index_map_out

    if custom_cuda: print("Warning: Custom CUDA path ignored for LNS mode."); custom_cuda = False

    assert(scale_bits > 0)

    # axes = [axes] if isinstance(axes, int) else list(axes)
    axes = [axes] if type(axes) == int else axes
    axes = [x + A.ndim if x < 0 else x for x in axes]
    if isinstance(elem_format, str):
        try:
            elem_format_obj = ElemFormat.from_str(elem_format)
        except Exception as e:
            print(f"ERROR converting elem_format string '{elem_format}': {e}")
            raise e
    elif isinstance(elem_format, ElemFormat):
        elem_format_obj = elem_format
    else:
        raise TypeError(f"elem_format must be string or ElemFormat enum, got {type(elem_format)}")
    
    # Shortcut for no quantization
    if elem_format == None:
        return A

    assert(scale_bits > 0)

    # Custom CUDA only supports limited rounding modes
    custom_cuda = custom_cuda and round in RoundingMode.string_enums()

    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)

    # Use quantize_mx_by_tile when there is only a single shared axis and
    # - The block size is small, OR
    # - The shared axis is not the innermost
    if A.device.type == "cuda" and custom_cuda and len(axes) == 1:
        axis = axes[0]
        if block_size == 0:
            block_size = A.shape[axis]

        if axis != len(A.shape) - 1 or block_size <= 32:
            A = A.contiguous()

            from . import custom_extensions as ce
            A = ce.funcs.quantize_mx_by_tile_func_cuda(
                A,
                scale_bits,
                ebits,
                mbits,
                max_norm,
                block_size,
                axis,
                flush_fp32_subnorms,
                RoundingMode[round],
            )
            return A


    # Perform tiling to the hardware vector size
    if block_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(
            A, axes, block_size
        )

    ####################
    # Quantize
    ####################
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

    if custom_cuda:
        # Custom CUDA code only supports a single axis
        if shared_exp_axes is None:
            axis = 0
        else:
            assert len(shared_exp_axes) == 1
            axis = shared_exp_axes[0]

        assert(shared_exp_method == "max")
        max_values = A.abs().max(dim=axis, keepdim=True).values

        A = A.contiguous()

        if A.device.type == "cuda":
            from . import custom_extensions as ce
            A = ce.funcs.quantize_mx_func_cuda(
                A, scale_bits, ebits, mbits, max_norm,
                max_values, axis,
                flush_fp32_subnorms, RoundingMode[round]);

        elif A.device.type == "cpu":
            from . import custom_extensions as ce
            A = ce.funcs.quantize_mx_func_cpp(
                A, scale_bits, ebits, mbits, max_norm,
                max_values, axis,
                flush_fp32_subnorms, RoundingMode[round]);

        else:
            raise ValueError("Unrecognized device type %s" % A.device.type)
    else:
        if lns_mode:
            A_original = A.clone()
            metric_to_optimize = "mse"
            
            candidate_formats = [
                (ElemFormat.lns4_e2m1, 2, 1)
            ]
            

            bias_values = [(b / 8.0) for b in range(-8, 8)]
            diff_values = [(b / 8.0) for b in range(-8, 8)]


            forced_block_shape: List[int]
            recalculated_axes: Optional[List[int]]
            # print("A.size()", A.size())
            if A.ndim >= 2:
                forced_block_shape = [A.shape[0], A.shape[1]]
                if A.ndim > 2:
                    recalculated_axes = list(range(2, A.ndim))
                else:
                    recalculated_axes = []

                block_shape = forced_block_shape
                re_axes = recalculated_axes

                # print("block_shape", block_shape)
                # print("re_axes", re_axes)

                A_best = torch.empty_like(A)
                best_bias_idx_map = torch.full(block_shape, -1, dtype=torch.long, device=A.device)
                best_diff_idx_map = torch.full(block_shape, -1, dtype=torch.long, device=A.device)
                best_format_idx_map = torch.full(block_shape, -1, dtype=torch.long, device=A.device)
                

                best_metric_map = torch.full(block_shape, -float("inf"), dtype=torch.float32, device=A.device)

                A_original_as_float = A_original.float()
                A_to_quant_float = A.float()
                
                epsilon = 1e-9
                cos_epsilon = 1e-8

                quantized_scale_pos_expanded: Optional[torch.Tensor] = None
                quantized_scale_neg_expanded: Optional[torch.Tensor] = None
                quantized_scale_sym_expanded: Optional[torch.Tensor] = None
                is_pos_mask: Optional[torch.Tensor] = None
                is_neg_mask: Optional[torch.Tensor] = None
                    
                block_max_abs_A_orig: torch.Tensor
                if re_axes is None: block_max_abs_A_orig = torch.max(torch.abs(A_to_quant_float))
                elif not re_axes: block_max_abs_A_orig = torch.abs(A_to_quant_float)
                else:
                    abs_A_val = torch.abs(A_to_quant_float)
                    temp_max_abs_iter_val = abs_A_val.clone()
                    for axis_idx_val in sorted(list(re_axes), reverse=True):
                        temp_max_abs_iter_val = torch.max(temp_max_abs_iter_val, dim=axis_idx_val, keepdim=False).values
                    block_max_abs_A_orig = temp_max_abs_iter_val
                
                block_max_abs_A_orig_safe = torch.clamp(block_max_abs_A_orig, min=epsilon)

                lns8_ebits_orig, lns8_mbits_orig = 5, 3 # test
                if verbose_lut: print(f"  Quantizing symmetric scale with LNS E{lns8_ebits_orig}M{lns8_mbits_orig}")
                lns8_base_lut_orig = _generate_base_lns_lut_for_core(lns8_mbits_orig, lns8_ebits_orig, A.device)
                
                quantized_scale_sym_orig: torch.Tensor
                if lns8_base_lut_orig.numel() > 1 or (lns8_base_lut_orig.numel()==1 and lns8_base_lut_orig[0]!=0.0):
                    q_block_max_abs_lin_orig = _snap_to_lut(block_max_abs_A_orig_safe, lns8_base_lut_orig)
                    quantized_scale_sym_orig = torch.clamp(q_block_max_abs_lin_orig, min=epsilon)
                else:
                    if verbose_lut: print("  Warning: Symm scale quant LUT invalid. Using original scale.")
                    quantized_scale_sym_orig = block_max_abs_A_orig_safe.clone()
                
                quantized_scale_sym_expanded = _expand_scale_like_A(quantized_scale_sym_orig, A_to_quant_float, re_axes)
                
                A_normalized = torch.where(torch.abs(quantized_scale_sym_expanded) > epsilon,
                                            A_to_quant_float / quantized_scale_sym_expanded,
                                            torch.zeros_like(A_to_quant_float))

            for fmt_idx, (fmt_enum, ebits_f, mbits_f) in enumerate(candidate_formats):
                current_m_other_bias_values = diff_values
                for bias_idx, bias_param_val in enumerate(bias_values):
                    for diff_idx, diff_param_val in enumerate(current_m_other_bias_values):

                        current_lut_orig_scale = _generate_biased_lns_lut_diff(
                            mbits_f, ebits_f, bias_param_val, diff_param_val, device=A.device,
                            verbose=(verbose_lut and fmt_idx==0 and bias_idx==0 and diff_idx==0)
                        )
                        if current_lut_orig_scale.numel() == 0 or (current_lut_orig_scale.numel()==1 and torch.all(current_lut_orig_scale==0.0)):
                            continue

                        current_lut_normalized = current_lut_orig_scale.clone()
                        if torch.any(current_lut_orig_scale != 0):
                            max_abs_lut = torch.max(torch.abs(current_lut_orig_scale))
                            if max_abs_lut > epsilon:
                                current_lut_normalized = current_lut_orig_scale / max_abs_lut
                        current_lut_normalized = torch.clamp(current_lut_normalized, -1.0, 1.0)

                        A_q_normalized = _snap_to_lut(A_normalized, current_lut_normalized)

                        A_dq = torch.zeros_like(A_q_normalized)
                        A_dq = A_q_normalized * quantized_scale_sym_expanded
                        
                        A_dq = torch.nan_to_num(A_dq, nan=0.0, posinf=0.0, neginf=0.0)
                        A_dq_float = A_dq.float()

                        metric_val_current_iter = None

                        if re_axes is not None:
                            block_dim_indices_metric = [i for i in range(A_dq_float.ndim) if i not in re_axes]
                            vector_dim_indices_metric = sorted(list(re_axes))
                            permute_order_metric = block_dim_indices_metric + vector_dim_indices_metric
                            
                            A_dq_permuted = A_dq_float.permute(*permute_order_metric)
                            A_orig_permuted = A_original_as_float.permute(*permute_order_metric)
                            
                            num_block_dims = len(block_dim_indices_metric)
                            if num_block_dims == 0:
                                A_dq_flat = A_dq_permuted.flatten().unsqueeze(0)
                                A_orig_flat = A_orig_permuted.flatten().unsqueeze(0)
                            else:
                                A_dq_flat = A_dq_permuted.flatten(start_dim=num_block_dims)
                                A_orig_flat = A_orig_permuted.flatten(start_dim=num_block_dims)

                            if metric_to_optimize == "cosine":
                                cos_sim = F.cosine_similarity(
                                    A_dq_flat, A_orig_flat, dim=-1, eps=cos_epsilon
                                )
                                metric_val_current_iter = cos_sim
                            elif metric_to_optimize == "mse":
                                mse_block = torch.mean((A_dq_flat - A_orig_flat) ** 2, dim=-1)
                                metric_val_current_iter = -mse_block
                            elif metric_to_optimize == "mae":
                                mae_block = torch.mean(torch.abs(A_dq_flat - A_orig_flat), dim=-1)
                                metric_val_current_iter = -mae_block
                            else:
                                raise ValueError(
                                    f"Unsupported metric_to_optimize: {metric_to_optimize}"
                                )

                        elif re_axes is None:
                            A_dq_flat_single_block = A_dq_float.flatten().unsqueeze(0)
                            A_orig_flat_single_block = A_original_as_float.flatten().unsqueeze(0)
                            if metric_to_optimize == "cosine":
                                cos_sim = F.cosine_similarity(A_dq_flat_single_block, A_orig_flat_single_block, dim=-1, eps=cos_epsilon)
                                metric_val_current_iter = cos_sim # shape: (1,)
                            elif metric_to_optimize == "mse":
                                mse_block = torch.mean((A_dq_flat_single_block - A_orig_flat_single_block)**2, dim=-1)
                                metric_val_current_iter = -mse_block # shape: (1,)
                            elif metric_to_optimize == "mae":
                                mae_block = torch.mean(torch.abs(A_dq_flat_single_block - A_orig_flat_single_block), dim=-1)
                                metric_val_current_iter = -mae_block # shape: (1,)
                            # ------------------
                            else:
                                raise ValueError(f"Unsupported metric_to_optimize: {metric_to_optimize}")
                        
                        elif not re_axes:
                            if metric_to_optimize == "cosine":
                                metric_val_current_iter = torch.zeros(block_shape, device=A.device, dtype=torch.float32)
                                # print("Warning: Cosine similarity for axes=() is ill-defined, using 0.")
                            elif metric_to_optimize == "mse":
                                mse_elementwise = (A_dq_float - A_original_as_float)**2
                                metric_val_current_iter = -mse_elementwise
                            elif metric_to_optimize == "mae":
                                mae_elementwise = torch.abs(A_dq_float - A_original_as_float)
                                metric_val_current_iter = -mae_elementwise
                            # ------------------
                            else:
                                raise ValueError(f"Unsupported metric_to_optimize: {metric_to_optimize}")
                        
                        if metric_val_current_iter is not None:
                            if metric_val_current_iter.shape != best_metric_map.shape:
                                raise ValueError(f"Shape mismatch: metric_val_current_iter ({metric_val_current_iter.shape}) vs best_metric_map ({best_metric_map.shape})")

                            better = metric_val_current_iter > best_metric_map
                            best_metric_map[better] = metric_val_current_iter[better]
                            best_format_idx_map[better] = fmt_idx
                            best_bias_idx_map[better] = bias_idx 
                            best_diff_idx_map[better] = diff_idx 

                            if torch.any(better):
                                update_mask_expanded = None
                                if re_axes is None: # better is scalar-like ([1])
                                    if better.all():
                                        A_best = A_dq.to(A_best.dtype)
                                elif not re_axes: # axes == (), better has A.shape
                                    A_dq_to_assign = A_dq.to(A_best.dtype)
                                    A_best[better] = A_dq_to_assign[better]
                                else:
                                    view_shape_for_better_mask = [1] * A.ndim; current_better_dim_idx = 0
                                    for i in range(A.ndim):
                                        if i not in re_axes:
                                            if current_better_dim_idx < better.ndim:
                                                view_shape_for_better_mask[i] = better.shape[current_better_dim_idx]
                                                current_better_dim_idx += 1
                                    
                                    update_mask_reshaped = better.view(view_shape_for_better_mask)
                                    final_update_mask = update_mask_reshaped.expand_as(A)
                                    A_dq_to_assign = A_dq.to(A_best.dtype)
                                    A_best[final_update_mask] = A_dq_to_assign[final_update_mask]
                
                not_updated_mask_block_level = (best_format_idx_map == -1)
                if torch.any(not_updated_mask_block_level):
                    fill_mask_expanded = None
                    if re_axes is None:
                        if not_updated_mask_block_level.all():
                            A_best = A_original.to(A_best.dtype)
                    elif not re_axes: # axes == ()
                        A_original_to_assign = A_original.to(A_best.dtype)
                        A_best[not_updated_mask_block_level] = A_original_to_assign[not_updated_mask_block_level]
                    else:
                        view_shape_for_fill = [1] * A.ndim; current_fill_dim_idx = 0
                        for i in range(A.ndim):
                            if i not in re_axes:
                                if current_fill_dim_idx < not_updated_mask_block_level.ndim:
                                    view_shape_for_fill[i] = not_updated_mask_block_level.shape[current_fill_dim_idx]
                                    current_fill_dim_idx += 1
                        
                        fill_mask_expanded_reshaped = not_updated_mask_block_level.view(view_shape_for_fill)
                        final_fill_mask = fill_mask_expanded_reshaped.expand_as(A_best)
                        A_original_to_assign = A_original.to(A_best.dtype)
                        A_best[final_fill_mask] = A_original_to_assign[final_fill_mask]


                A = A_best
            
            # TODO
            else:
                candidate_formats = [
                    (ElemFormat.lns4_e2m1, 2, 1)
                ]

                bias_values = [0]
                diff_values = [0]

                forced_block_shape: List[int]
                recalculated_axes: Optional[List[int]]

                
                if A.ndim >= 2:
                    forced_block_shape = [A.shape[1], A.shape[2]]
                    if A.ndim > 2:
                        recalculated_axes = list(range(3, A.ndim))
                        
                    else:
                        recalculated_axes = []

                    block_shape = forced_block_shape
                    re_axes = recalculated_axes

                    A_original_as_float = A_original.float()
                    A_to_quant_float = A.float()
                    
                    epsilon = 1e-9
                    cos_epsilon = 1e-8

                    quantized_scale_pos_expanded: Optional[torch.Tensor] = None
                    quantized_scale_neg_expanded: Optional[torch.Tensor] = None
                    quantized_scale_sym_expanded: Optional[torch.Tensor] = None
                    is_pos_mask: Optional[torch.Tensor] = None
                    is_neg_mask: Optional[torch.Tensor] = None

                    block_max_abs_A_orig: torch.Tensor
                    if re_axes is None: block_max_abs_A_orig = torch.max(torch.abs(A_to_quant_float))
                    elif not re_axes: block_max_abs_A_orig = torch.abs(A_to_quant_float)
                    else:
                        abs_A_val = torch.abs(A_to_quant_float)
                        temp_max_abs_iter_val = abs_A_val.clone()
                        for axis_idx_val in sorted(list(re_axes), reverse=True):
                            temp_max_abs_iter_val = torch.max(temp_max_abs_iter_val, dim=axis_idx_val, keepdim=False).values
                        block_max_abs_A_orig = temp_max_abs_iter_val
                    
                    block_max_abs_A_orig_safe = torch.clamp(block_max_abs_A_orig, min=epsilon)

                    lns8_ebits_orig, lns8_mbits_orig = 5, 3 # test
                    if verbose_lut: print(f"  Quantizing symmetric scale with LNS E{lns8_ebits_orig}M{lns8_mbits_orig}")
                    lns8_base_lut_orig = _generate_base_lns_lut_for_core(lns8_mbits_orig, lns8_ebits_orig, A.device)
                    
                    quantized_scale_sym_orig: torch.Tensor
                    if lns8_base_lut_orig.numel() > 1 or (lns8_base_lut_orig.numel()==1 and lns8_base_lut_orig[0]!=0.0):
                        q_block_max_abs_lin_orig = _snap_to_lut(block_max_abs_A_orig_safe, lns8_base_lut_orig)
                        quantized_scale_sym_orig = torch.clamp(q_block_max_abs_lin_orig, min=epsilon)
                    else:
                        if verbose_lut: print("  Warning: Symm scale quant LUT invalid. Using original scale.")
                        quantized_scale_sym_orig = block_max_abs_A_orig_safe.clone()
                    
                    quantized_scale_sym_expanded = _expand_scale_like_A(quantized_scale_sym_orig, A_to_quant_float, re_axes)
                    
                    A_normalized = torch.where(torch.abs(quantized_scale_sym_expanded) > epsilon,
                                                A_to_quant_float / quantized_scale_sym_expanded,
                                                torch.zeros_like(A_to_quant_float))
                
                for fmt_idx, (fmt_enum, ebits_f, mbits_f) in enumerate(candidate_formats):
                    current_m_other_bias_values = diff_values
                    
                    for bias_idx, bias_param_val in enumerate(bias_values):
                        for diff_idx, diff_param_val in enumerate(current_m_other_bias_values):

                            current_lut_orig_scale = _generate_biased_lns_lut_diff(
                                mbits_f, ebits_f, bias_param_val, diff_param_val, device=A.device,
                                verbose=(verbose_lut and fmt_idx==0 and bias_idx==0 and diff_idx==0)
                            )
                            if current_lut_orig_scale.numel() == 0 or (current_lut_orig_scale.numel()==1 and torch.all(current_lut_orig_scale==0.0)):
                                continue

                            current_lut_normalized = current_lut_orig_scale.clone()
                            if torch.any(current_lut_orig_scale != 0):
                                max_abs_lut = torch.max(torch.abs(current_lut_orig_scale))
                                if max_abs_lut > epsilon:
                                    current_lut_normalized = current_lut_orig_scale / max_abs_lut
                            current_lut_normalized = torch.clamp(current_lut_normalized, -1.0, 1.0)

                            A_q_normalized = _snap_to_lut(A_normalized, current_lut_normalized)

                            A_dq = torch.zeros_like(A_q_normalized)
                            A_dq = A_q_normalized * quantized_scale_sym_expanded
                            A_dq = torch.nan_to_num(A_dq, nan=0.0, posinf=0.0, neginf=0.0)
                            A_dq_float = A_dq.float()

                A = A_dq_float

        else: # lns_mode == False
            shared_exp = _shared_exponents(
                A, method=shared_exp_method, axes=shared_exp_axes, ebits=0,
            )

            # Flush subnormal FP32 inputs to zero
            if flush_fp32_subnorms:
                A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

            # Offset the max exponent by the largest representable exponent
            # in the element data format
            shared_exp = shared_exp - emax

            scale_emax = 2**(scale_bits-1) - 1
            shared_exp[shared_exp > scale_emax] = float("NaN")
            shared_exp[shared_exp < -scale_emax] = -scale_emax

            A = A / (2**shared_exp)

            A = _quantize_elemwise_core(
                    A, mbits, ebits, max_norm, round=round,
                    allow_denorm=True, saturate_normals=True,
                    custom_cuda=custom_cuda)

            A = A * (2**shared_exp)

        
        if block_size > 0:
            A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

        # Return the quantized tensor.
        return A

def _quantize_mx_A(
    A,
    scale_bits,
    elem_format,    # can be None for no quantization
    shared_exp_method="max",
    axes=None,
    block_size=0,
    round="nearest",
    flush_fp32_subnorms=False,
    custom_cuda=False,
    lns_mode=False,
    verbose_lut: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Quantize matrix A in MX format."""

    best_bias_index_map_out: Optional[torch.Tensor] = None

    if elem_format is None: return A, best_bias_index_map_out

    if custom_cuda: print("Warning: Custom CUDA path ignored for LNS mode."); custom_cuda = False

    assert(scale_bits > 0)

    # axes = [axes] if isinstance(axes, int) else list(axes)
    axes = [axes] if type(axes) == int else axes
    axes = [x + A.ndim if x < 0 else x for x in axes]
    if isinstance(elem_format, str):
        try:
            elem_format_obj = ElemFormat.from_str(elem_format)
        except Exception as e:
            print(f"ERROR converting elem_format string '{elem_format}': {e}")
            raise e
    elif isinstance(elem_format, ElemFormat):
        elem_format_obj = elem_format
    else:
        raise TypeError(f"elem_format must be string or ElemFormat enum, got {type(elem_format)}")
    
    # Shortcut for no quantization
    if elem_format == None:
        return A

    assert(scale_bits > 0)

    # Custom CUDA only supports limited rounding modes
    custom_cuda = custom_cuda and round in RoundingMode.string_enums()

    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)

    # Use quantize_mx_by_tile when there is only a single shared axis and
    # - The block size is small, OR
    # - The shared axis is not the innermost
    if A.device.type == "cuda" and custom_cuda and len(axes) == 1:
        axis = axes[0]
        if block_size == 0:
            block_size = A.shape[axis]

        if axis != len(A.shape) - 1 or block_size <= 32:
            A = A.contiguous()

            from . import custom_extensions as ce
            A = ce.funcs.quantize_mx_by_tile_func_cuda(
                A,
                scale_bits,
                ebits,
                mbits,
                max_norm,
                block_size,
                axis,
                flush_fp32_subnorms,
                RoundingMode[round],
            )
            return A


    # Perform tiling to the hardware vector size
    if block_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(
            A, axes, block_size
        )

    ####################
    # Quantize
    ####################
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

    
    if lns_mode:
        A_original = A.clone()
        A_to_quant_float = A.float()
        epsilon = 1e-9
        candidate_formats = [(ElemFormat.lns4_e2m1, 2, 1)]
        block_axis = -1 
        re_axes = [block_axis]

        diff: torch.Tensor
        A_normalized: torch.Tensor
        
        # ==========================================================================
        # ==========================================================================
        # 1. Find Top-3 absolute values
        abs_A = torch.abs(A_to_quant_float)
        top3_abs, _ = torch.topk(abs_A, k=3, dim=block_axis)

        v1 = torch.select(top3_abs, block_axis, 0)  # max
        v2 = torch.select(top3_abs, block_axis, 1)
        v3 = torch.select(top3_abs, block_axis, 2)

        # 2. Convert to log scale
        epsilon = 1e-9
        exp1 = torch.log2(v1 + epsilon)
        exp2 = torch.log2(v2 + epsilon)
        exp3 = torch.log2(v3 + epsilon)

        # 3. Directly calculate bias and diff from the Top-3 log values
        # This corresponds to the 'cond_none' case from the original algorithm
        bias_val = torch.where(
            exp1 - exp2 >= 0.5,
            torch.clamp(exp1 - exp2 - 0.5, min=0, max=32.0),
            torch.zeros_like(exp1)
        )
        diff_val = torch.where(
            exp2 - exp3 >= 0.5,
            torch.clamp(exp2 - exp3 - 0.5, min=0, max=32.0),
            torch.zeros_like(exp1)
        )

        b_lns_e, b_lns_m = 4, 1
        b_lut_list = [0.0,0.5,1.0,2.0,3.0,4.0,6.0,8.0]
        b_lut = torch.tensor(b_lut_list, device=bias_val.device, dtype=bias_val.dtype)
        
        bias_val_q = bias_val
        diff_val_q = diff_val

        bias = _expand_scale_like_A(bias_val_q, A_to_quant_float, re_axes)
        diff = _expand_scale_like_A(diff_val_q, A_to_quant_float, re_axes)

        s_lns_e, s_lns_m = 5, 3 # test
        s_lut = _generate_base_lns_lut_for_core(s_lns_m, s_lns_e, A.device)
        quantized_scale = _snap_to_lut(torch.clamp(v1, min=epsilon), s_lut)
        quantized_scale_expanded = _expand_scale_like_A(quantized_scale, A_to_quant_float, re_axes)
        A_normalized = torch.where(
            torch.abs(quantized_scale_expanded) > epsilon,
            A_to_quant_float / quantized_scale_expanded,
            torch.zeros_like(A_to_quant_float)
        )
        A_normalized = torch.nan_to_num(A_normalized, nan=0.0, posinf=0.0, neginf=0.0)
            
        # ==========================================================================
        # ==========================================================================
        
        dims = list(range(A_normalized.ndim))
        positive_block_axis = dims[block_axis]
        permute_order = [d for d in dims if d != positive_block_axis] + [positive_block_axis]
        
        A_permuted = A_normalized.permute(*permute_order)
        bias_permuted = bias.permute(*permute_order)
        diff_permuted = diff.permute(*permute_order)
        
        scale_permuted = quantized_scale_expanded.permute(*permute_order)

        num_groups = A_permuted.shape[:-1].numel()
        block_size = A_permuted.shape[-1]

        # Flatten
        A_flat = A_permuted.contiguous().view(num_groups, block_size)
        bias_flat = bias_permuted[..., 0].flatten()
        diff_flat = diff_permuted[..., 0].flatten()
        
        scale_flat = scale_permuted.contiguous().view(num_groups, block_size)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        chunk_size = 65536 * 3 
        A_dq_flat = torch.zeros_like(A_flat)

        for i in range(0, num_groups, chunk_size):
            end_idx = min(i + chunk_size, num_groups)

            A_chunk = A_flat[i:end_idx]
            bias_chunk = bias_flat[i:end_idx].unsqueeze(-1)
            diff_chunk = diff_flat[i:end_idx].unsqueeze(-1)
            
            fmt_enum, ebits_f, mbits_f = candidate_formats[0]
            
            lut_batch_chunk = _generate_biased_lns_lut_diff_activation(
                mbits_f, ebits_f, bias_chunk, diff_chunk, device=A.device
            )
            
            max_abs_lut, _ = torch.max(torch.abs(lut_batch_chunk), dim=-1, keepdim=True)
            lut_batch_normalized = torch.where(max_abs_lut > epsilon, lut_batch_chunk / max_abs_lut, lut_batch_chunk)
            
            A_q_chunk = _snap_to_lut_activation(A_chunk, lut_batch_normalized)

            scale_chunk = scale_flat[i:end_idx]
            A_dq_chunk = A_q_chunk * scale_chunk
            
            A_dq_flat[i:end_idx] = A_dq_chunk

        final_shape = A_normalized.permute(*permute_order).shape
        A_dq_permuted = A_dq_flat.reshape(final_shape)
        inverse_permute_order = np.argsort(permute_order)
        A_dq = A_dq_permuted.permute(*inverse_permute_order)
        
        A = torch.nan_to_num(A_dq.float())

        
    else: # lns_mode == False
            shared_exp = _shared_exponents(
                A, method=shared_exp_method, axes=shared_exp_axes, ebits=0,
            )

            # Flush subnormal FP32 inputs to zero
            if flush_fp32_subnorms:
                A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

            # Offset the max exponent by the largest representable exponent
            # in the element data format
            shared_exp = shared_exp - emax

            scale_emax = 2**(scale_bits-1) - 1
            shared_exp[shared_exp > scale_emax] = float("NaN")
            shared_exp[shared_exp < -scale_emax] = -scale_emax

            A = A / (2**shared_exp)

            A = _quantize_elemwise_core(
                    A, mbits, ebits, max_norm, round=round,
                    allow_denorm=True, saturate_normals=True,
                    custom_cuda=custom_cuda)

            A = A * (2**shared_exp)

        
    if block_size > 0:
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    # Return the quantized matrix.
    return A

def _quantize_mx_B(
    A,
    scale_bits,
    elem_format,    # can be None for no quantization
    shared_exp_method="max",
    axes=None,
    block_size=0,
    round="nearest",
    flush_fp32_subnorms=False,
    custom_cuda=False,
    lns_mode=False,
    verbose_lut: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Quantize matrix B in MX format."""

    best_bias_index_map_out: Optional[torch.Tensor] = None

    if elem_format is None: return A, best_bias_index_map_out

    if custom_cuda: print("Warning: Custom CUDA path ignored for LNS mode."); custom_cuda = False

    assert(scale_bits > 0)

    # axes = [axes] if isinstance(axes, int) else list(axes)
    axes = [axes] if type(axes) == int else axes
    axes = [x + A.ndim if x < 0 else x for x in axes]
    if isinstance(elem_format, str):
        try:
            elem_format_obj = ElemFormat.from_str(elem_format)
        except Exception as e:
            print(f"ERROR converting elem_format string '{elem_format}': {e}")
            raise e
    elif isinstance(elem_format, ElemFormat):
        elem_format_obj = elem_format
    else:
        raise TypeError(f"elem_format must be string or ElemFormat enum, got {type(elem_format)}")
    
    # Shortcut for no quantization
    if elem_format == None:
        return A

    assert(scale_bits > 0)

    # Custom CUDA only supports limited rounding modes
    custom_cuda = custom_cuda and round in RoundingMode.string_enums()

    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)

    # Use quantize_mx_by_tile when there is only a single shared axis and
    # - The block size is small, OR
    # - The shared axis is not the innermost
    if A.device.type == "cuda" and custom_cuda and len(axes) == 1:
        axis = axes[0]
        if block_size == 0:
            block_size = A.shape[axis]

        if axis != len(A.shape) - 1 or block_size <= 32:
            A = A.contiguous()

            from . import custom_extensions as ce
            A = ce.funcs.quantize_mx_by_tile_func_cuda(
                A,
                scale_bits,
                ebits,
                mbits,
                max_norm,
                block_size,
                axis,
                flush_fp32_subnorms,
                RoundingMode[round],
            )
            return A


    # Perform tiling to the hardware vector size
    if block_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(
            A, axes, block_size
        )

    ####################
    # Quantize
    ####################
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

    
    if lns_mode:
        A_original = A.clone()
        A_to_quant_float = A.float()
        epsilon = 1e-9
        candidate_formats = [(ElemFormat.lns4_e2m1, 2, 1)]
        block_axis = -2 
        re_axes = [block_axis]

        diff: torch.Tensor
        A_normalized: torch.Tensor
        
        # ==========================================================================
        # ==========================================================================
        # 1. Find Top-3 absolute values
        abs_A = torch.abs(A_to_quant_float)
        top3_abs, _ = torch.topk(abs_A, k=3, dim=block_axis)

        v1 = torch.select(top3_abs, block_axis, 0)  # max
        v2 = torch.select(top3_abs, block_axis, 1)
        v3 = torch.select(top3_abs, block_axis, 2)

        # 2. Convert to log scale
        epsilon = 1e-9
        exp1 = torch.log2(v1 + epsilon)
        exp2 = torch.log2(v2 + epsilon)
        exp3 = torch.log2(v3 + epsilon)

        # 3. Directly calculate bias and diff from the Top-3 log values
        # This corresponds to the 'cond_none' case from the original algorithm
        bias_val = torch.where(
            exp1 - exp2 >= 0.5,
            torch.clamp(exp1 - exp2 - 0.5, min=0, max=32.0),
            torch.zeros_like(exp1)
        )
        diff_val = torch.where(
            exp2 - exp3 >= 0.5,
            torch.clamp(exp2 - exp3 - 0.5, min=0, max=32.0),
            torch.zeros_like(exp1)
        )

        b_lns_e, b_lns_m = 4, 1
        b_lut_list = [0.0,0.5,1.0,2.0,3.0,4.0,6.0,8.0]
        b_lut = torch.tensor(b_lut_list, device=bias_val.device, dtype=bias_val.dtype)
        
        bias_val_q = bias_val
        diff_val_q = diff_val

        bias = _expand_scale_like_A(bias_val_q, A_to_quant_float, re_axes)
        diff = _expand_scale_like_A(diff_val_q, A_to_quant_float, re_axes)

        s_lns_e, s_lns_m = 5, 3 # test
        s_lut = _generate_base_lns_lut_for_core(s_lns_m, s_lns_e, A.device)
        quantized_scale = _snap_to_lut(torch.clamp(v1, min=epsilon), s_lut)
        quantized_scale_expanded = _expand_scale_like_A(quantized_scale, A_to_quant_float, re_axes)
        A_normalized = torch.where(
            torch.abs(quantized_scale_expanded) > epsilon,
            A_to_quant_float / quantized_scale_expanded,
            torch.zeros_like(A_to_quant_float)
        )
        A_normalized = torch.nan_to_num(A_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        # ==========================================================================
        # ==========================================================================
        
        dims = list(range(A_normalized.ndim))
        positive_block_axis = dims[block_axis]
        permute_order = [d for d in dims if d != positive_block_axis] + [positive_block_axis]
        
        A_permuted = A_normalized.permute(*permute_order)
        bias_permuted = bias.permute(*permute_order)
        diff_permuted = diff.permute(*permute_order)
        
        scale_permuted = quantized_scale_expanded.permute(*permute_order)

        num_groups = A_permuted.shape[:-1].numel()
        block_size = A_permuted.shape[-1]

        # Flatten
        A_flat = A_permuted.contiguous().view(num_groups, block_size)
        bias_flat = bias_permuted[..., 0].flatten()
        diff_flat = diff_permuted[..., 0].flatten()
        
        scale_flat = scale_permuted.contiguous().view(num_groups, block_size)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        chunk_size = 65536 * 3 
        A_dq_flat = torch.zeros_like(A_flat)

        for i in range(0, num_groups, chunk_size):
            end_idx = min(i + chunk_size, num_groups)

            A_chunk = A_flat[i:end_idx]
            bias_chunk = bias_flat[i:end_idx].unsqueeze(-1)
            diff_chunk = diff_flat[i:end_idx].unsqueeze(-1)
            
            fmt_enum, ebits_f, mbits_f = candidate_formats[0]
            
            lut_batch_chunk = _generate_biased_lns_lut_diff_activation(
                mbits_f, ebits_f, bias_chunk, diff_chunk, device=A.device
            )
            
            max_abs_lut, _ = torch.max(torch.abs(lut_batch_chunk), dim=-1, keepdim=True)
            lut_batch_normalized = torch.where(max_abs_lut > epsilon, lut_batch_chunk / max_abs_lut, lut_batch_chunk)
            
            A_q_chunk = _snap_to_lut_activation(A_chunk, lut_batch_normalized)

            scale_chunk = scale_flat[i:end_idx]
            A_dq_chunk = A_q_chunk * scale_chunk
            
            A_dq_flat[i:end_idx] = A_dq_chunk

        final_shape = A_normalized.permute(*permute_order).shape
        A_dq_permuted = A_dq_flat.reshape(final_shape)
        inverse_permute_order = np.argsort(permute_order)
        A_dq = A_dq_permuted.permute(*inverse_permute_order)
        
        A = torch.nan_to_num(A_dq.float())


    else: # lns_mode == False
            shared_exp = _shared_exponents(
                A, method=shared_exp_method, axes=shared_exp_axes, ebits=0,
            )

            # Flush subnormal FP32 inputs to zero
            if flush_fp32_subnorms:
                A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

            # Offset the max exponent by the largest representable exponent
            # in the element data format
            shared_exp = shared_exp - emax

            scale_emax = 2**(scale_bits-1) - 1
            shared_exp[shared_exp > scale_emax] = float("NaN")
            shared_exp[shared_exp < -scale_emax] = -scale_emax

            A = A / (2**shared_exp)

            A = _quantize_elemwise_core(
                    A, mbits, ebits, max_norm, round=round,
                    allow_denorm=True, saturate_normals=True,
                    custom_cuda=custom_cuda)

            A = A * (2**shared_exp)

        
    if block_size > 0:
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    # Return the quantized matrix.
    return A


def quantize_mx_op(
    A,
    mx_specs: dict,
    elem_format=None,
    block_size=None,
    axes=None,
    round="nearest",
    expand_and_reshape=False,
):
    mx_assert_test(mx_specs)

    if elem_format == None:
        return A
    elif type(elem_format) is str:
        elem_format = ElemFormat.from_str(elem_format)

    if block_size == None:
        block_size = mx_specs["block_size"]

    if mx_specs["scale_bits"] == 0:
        scale_bits = 8
    else:
        scale_bits = mx_specs["scale_bits"]

    return _quantize_mx(
            A, scale_bits,
            elem_format, block_size=block_size,
            axes=axes, round=round,
            shared_exp_method=mx_specs["shared_exp_method"],
            flush_fp32_subnorms=mx_specs["mx_flush_fp32_subnorms"],
            custom_cuda=mx_specs["custom_cuda"],
            lns_mode=mx_specs["lns_mode"],
            )

def quantize_mx_op_matmul_A(
    A,
    mx_specs: dict,
    elem_format=None,
    block_size=None,
    axes=None,
    round="nearest",
    expand_and_reshape=False,
):
    mx_assert_test(mx_specs)

    if elem_format == None:
        return A
    elif type(elem_format) is str:
        elem_format = ElemFormat.from_str(elem_format)

    if block_size == None:
        block_size = mx_specs["block_size"]

    if mx_specs["scale_bits"] == 0:
        scale_bits = 8
    else:
        scale_bits = mx_specs["scale_bits"]

    return _quantize_mx_A(
            A, scale_bits,
            elem_format, block_size=block_size,
            axes=axes, round=round,
            shared_exp_method=mx_specs["shared_exp_method"],
            flush_fp32_subnorms=mx_specs["mx_flush_fp32_subnorms"],
            custom_cuda=mx_specs["custom_cuda"],
            lns_mode=mx_specs["lns_mode"],
            )

def quantize_mx_op_matmul_B(
    A,
    mx_specs: dict,
    elem_format=None,
    block_size=None,
    axes=None,
    round="nearest",
    expand_and_reshape=False,
):
    mx_assert_test(mx_specs)

    if elem_format == None:
        return A
    elif type(elem_format) is str:
        elem_format = ElemFormat.from_str(elem_format)

    if block_size == None:
        block_size = mx_specs["block_size"]

    if mx_specs["scale_bits"] == 0:
        scale_bits = 8
    else:
        scale_bits = mx_specs["scale_bits"]

    return _quantize_mx_B(
            A, scale_bits,
            elem_format, block_size=block_size,
            axes=axes, round=round,
            shared_exp_method=mx_specs["shared_exp_method"],
            flush_fp32_subnorms=mx_specs["mx_flush_fp32_subnorms"],
            custom_cuda=mx_specs["custom_cuda"],
            lns_mode=mx_specs["lns_mode"],
            )


