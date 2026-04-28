"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Name:    elemwise_ops.py

Pytorch functions for elementwise (i.e. bfloat) quantization.

Usage Notes:
 - Use the "Exposed Methods" below to implement autograd functions
 - Use autograd functions to then implement torch.nn.Module(s)
 - Do *not* use methods in this file in Modules, they have no defined
   backwards pass and will block gradient computation.
 - Avoid importing internal function if at all possible.

Exposed Methods:
    quantize_elemwise_op - quantizes a tensor to bfloat or other
                           custom float format
"""
import torch
import math
from .formats import RoundingMode, _get_format_params
from .formats import _get_min_norm, _get_max_norm
from typing import Optional, Tuple, List, Dict ,  Any

# -------------------------------------------------------------------------
# Helper funcs
# -------------------------------------------------------------------------
# Never explicitly compute 2**(-exp) since subnorm numbers have
# exponents smaller than -126
def _safe_lshift(x, bits, exp):
    if exp is None:
        return x * (2**bits)
    else:
        return x / (2 ** exp) * (2**bits)


def _safe_rshift(x, bits, exp):
    if exp is None:
        return x / (2**bits)
    else:
        return x / (2**bits) * (2 ** exp)


def _round_mantissa(A, bits, round, clamp=False):
    """
    Rounds mantissa to nearest bits depending on the rounding method 'round'
    Args:
      A     {PyTorch tensor} -- Input tensor
      round {str}            --  Rounding method
                                 "floor" rounds to the floor
                                 "nearest" rounds to ceil or floor, whichever is nearest
    Returns:
      A {PyTorch tensor} -- Tensor with mantissas rounded
    """

    if round == "dither":
        rand_A = torch.rand_like(A, requires_grad=False)
        A = torch.sign(A) * torch.floor(torch.abs(A) + rand_A)
    elif round == "floor":
        A = torch.sign(A) * torch.floor(torch.abs(A))
    elif round == "nearest":
        A = torch.sign(A) * torch.floor(torch.abs(A) + 0.5)
    elif round == "even":
        absA = torch.abs(A)
        # find 0.5, 2.5, 4.5 ...
        maskA = ((absA - 0.5) % 2 == torch.zeros_like(A)).type(A.dtype)
        A = torch.sign(A) * (torch.floor(absA + 0.5) - maskA)
    else:
        raise Exception("Unrecognized round method %s" % (round))

    # Clip values that cannot be expressed by the specified number of bits
    if clamp:
        max_mantissa = 2 ** (bits - 1) - 1
        A = torch.clamp(A, -max_mantissa, max_mantissa)
    return A


# -------------------------------------------------------------------------
# Main funcs
# -------------------------------------------------------------------------
def quantize_lns_with_lut(A_norm, lut, round='nearest'):
    """
    Quantize normalized values by snapping them to the nearest LUT entry.

    Args:
        A_norm (Tensor): Normalized tensor values.
        lut (Tensor): LNS lookup table.
        round (str): Rounding mode. Only `nearest` is supported.

    Returns:
        Tensor: Quantized tensor.
    """
    if round != 'nearest':
        raise NotImplementedError("Only 'nearest' rounding is implemented for LUT quantization.")

    A_flat = A_norm.reshape(-1).to(torch.float32)
    lut_f32 = lut.to(torch.float32)
    dists = torch.abs(A_flat.unsqueeze(1) - lut_f32.unsqueeze(0))

    indices = torch.argmin(dists, dim=1)

    snapped_flat = lut_f32[indices]

    return snapped_flat.view_as(A_norm).to(A_norm.dtype)

# Standard LNS LUT quantization without extra bias terms.
def _quantize_elemwise_lns_core_standard(A, bits, exp_bits, max_norm, round='nearest',
                                        saturate_normals=False, allow_denorm=True,
                                        custom_cuda=False, lns_mode=False, verbose=False):
    def snap_to_lut(tensor_in, lut):
        tensor_flat = tensor_in.reshape(-1).to(torch.float32)
        snapped = torch.empty_like(tensor_flat)
        chunk_size = 32768 * 4
        used_indices = []
        for start in range(0, tensor_flat.shape[0], chunk_size):
            end = start + chunk_size
            chunk = tensor_flat[start:end].unsqueeze(1)
            dists = torch.abs(chunk - lut.unsqueeze(0))
            idx = torch.argmin(dists, dim=1)
            snapped[start:end] = lut[idx]
            used_indices.append(idx)
        if verbose:
            all_used_indices = torch.cat(used_indices)
            unique_used_indices = torch.unique(all_used_indices)
            num_unique_used = unique_used_indices.numel()
            print(f"    [LUT Stats] LUT Size: {lut.numel()}, Used Entries: {num_unique_used} ({num_unique_used/lut.numel()*100:.2f}%)")
            if num_unique_used < 20:
                 print(f"    [LUT Stats] Used Values: {lut[unique_used_indices].cpu().numpy()}")

        return snapped.view_as(tensor_in)

    emax = 2 ** (exp_bits - 1) - 1 if exp_bits > 0 else 0
    emin = -emax - 1 if exp_bits > 0 else 0
    mvals = [i / (2 ** bits) for i in range(2 ** bits)]
    lns_vals = [0.0]
    for s in [1.0, -1.0]:
        for e in range(int(emin), int(emax) + 1):
            for m in mvals:
                if e == emin and m == 0: continue
                try:
                    val = s * (2.0 ** (e + m))
                    if math.isnan(val) or math.isinf(val): continue
                except OverflowError: continue
                lns_vals.append(val)

    unique_lns_vals = sorted(list(set(lns_vals)))
    if not unique_lns_vals:
        if verbose: print("    Error: Standard LNS LUT generation failed.")
        lns_lut = torch.tensor([0.0], dtype=torch.float32, device=A.device)
    else:
        if 0.0 not in unique_lns_vals:
             if verbose: print("    Warning: 0.0 missing from standard LUT, adding.")
             unique_lns_vals.insert(0, 0.0)
             unique_lns_vals = sorted(list(set(unique_lns_vals)))
        lns_lut = torch.tensor(unique_lns_vals, dtype=torch.float32, device=A.device)

    if verbose:
        print(f"    Info [Standard LUT]: Generated. Size={lns_lut.numel()}. "
              f"Range=[{lns_lut.min():.4f}, {lns_lut.max():.4f}]")

    out = snap_to_lut(A, lns_lut)
    return out.to(A.dtype)

def _generate_base_lns_lut_for_core(bits: int, exp_bits: int, device: torch.device) -> torch.Tensor:
    """Generate a base LNS LUT with no bias or diff terms."""
    emax = 2 ** (exp_bits - 1) - 1 if exp_bits > 0 else 0
    emin = -emax - 1 if exp_bits > 0 else 0
    mvals = [i / (2 ** bits) for i in range(2 ** bits)]
    lns_vals_set = {0.0}
    for s in [1.0, -1.0]:
        for e in range(int(emin), int(emax) + 1):
            for m in mvals:
                if exp_bits > 0 and e == emin and m == 0.0: continue
                try:
                    val = s * (2.0 ** (e + m))
                    if not math.isfinite(val): continue
                    lns_vals_set.add(val)
                except OverflowError: continue
    final_lut = torch.tensor(sorted(list(lns_vals_set)), dtype=torch.float32, device=device)
    return final_lut if final_lut.numel() > 0 and not (final_lut.numel()==1 and final_lut[0]==0.0) else torch.tensor([0.0], dtype=torch.float32, device=device)

def _quantize_elemwise_lns_core_sf_max_mapping(
    A: torch.Tensor,
    bits: int,
    exp_bits: int,
    axes: Tuple[int, ...] = None,
    # max_norm: float,
    # round_mode='nearest',
    # allow_denorm=True,
    # custom_cuda=False,
    # lns_mode=False
):
    """
    Quantize with an LNS LUT after scaling each block by its maximum magnitude.
    """
    A_float = A.float()
    epsilon = 1e-9

    base_lns_lut = _generate_base_lns_lut_for_core(bits, exp_bits, A.device)

    if base_lns_lut.numel() == 0 or \
       (base_lns_lut.numel() == 1 and torch.all(base_lns_lut == 0.0)):
        return torch.zeros_like(A)

    abs_lut_values = torch.abs(base_lns_lut)
    non_zero_abs_lut = abs_lut_values[abs_lut_values > epsilon]
    if non_zero_abs_lut.numel() > 0:
        lut_max_abs = torch.max(non_zero_abs_lut)
    else:
        lut_max_abs = torch.tensor(epsilon, device=A.device, dtype=torch.float32)
    lut_max_abs_safe = lut_max_abs.clamp(min=epsilon)
    block_max_abs_A = None
    if axes is None:
        block_max_abs_A = torch.max(torch.abs(A_float))
    else:
        block_max_abs_A = torch.amax(torch.abs(A_float), dim=axes, keepdim=True)
    
    block_max_abs_A_safe = torch.clamp(block_max_abs_A, min=epsilon)

    scaling_factor = torch.where(block_max_abs_A_safe > epsilon,
                                 lut_max_abs_safe / block_max_abs_A_safe,
                                 torch.ones_like(block_max_abs_A_safe))

    A_scaled = A_float * scaling_factor

    A_flat_scaled = A_scaled.reshape(-1)
    snapped_scaled_vals = torch.empty_like(A_flat_scaled)
    chunk_size = 32768 * 4
    
    for start in range(0, A_flat_scaled.shape[0], chunk_size):
        end = start + chunk_size
        chunk = A_flat_scaled[start:end].unsqueeze(1)
        dists = torch.abs(chunk - base_lns_lut)
        idx = torch.argmin(dists, dim=1)
        snapped_scaled_vals[start:end] = base_lns_lut[idx]
    
    A_q_scaled = snapped_scaled_vals.view_as(A_float)

    A_dequant = torch.where(torch.abs(scaling_factor) > epsilon,
                            A_q_scaled / scaling_factor,
                            torch.zeros_like(A_q_scaled))
    
    A_dequant = torch.nan_to_num(A_dequant, nan=0.0, posinf=0.0, neginf=0.0)

    return A_dequant.to(A.dtype)

def _quantize_elemwise_lns_core(A, bits, exp_bits, max_norm, round='nearest',
                                saturate_normals=False, allow_denorm=True,
                                custom_cuda=False, lns_mode=False):
    """
    Quantize a tensor by snapping it directly to an LNS LUT.
    """
    eps = 1e-6
    A_abs = torch.clamp(torch.abs(A), min=eps)
    sign = torch.sign(A)

    # Step 1: build the full LNS LUT.
    emax = 2 ** (exp_bits - 1) - 1
    if exp_bits == 0:
        emin = 0
    else:
        emin = -emax-1
    mvals = [i / (2 ** bits) for i in range(2 ** bits)]

    lns_vals = [0.0]

    for s in [1.0, -1.0]:
        for e in range(int(emin), int(emax) + 1):
            for m in mvals:
                if e == emin and m == 0:
                    continue
                val = s * (2 ** (e + m))
                lns_vals.append(val)

    lns_lut = torch.tensor(sorted(set(lns_vals)), dtype=torch.float32, device=A.device)

    # Step 2: snap the tensor to the LUT.
    A_flat = A.reshape(-1).to(torch.float32)
    snapped = torch.empty_like(A_flat)
    chunk_size = 32768 * 4

    used_lut_indices = []
    
    for start in range(0, A_flat.shape[0], chunk_size):
        end = start + chunk_size
        chunk = A_flat[start:end].unsqueeze(1)
        dists = torch.abs(chunk - lns_lut.unsqueeze(0))
        idx = torch.argmin(dists, dim=1)
        snapped[start:end] = lns_lut[idx]
        used_lut_indices.append(idx)

    out = snapped.view_as(A)

    return out.to(A.dtype)

def _quantize_elemwise_core(A, bits, exp_bits, max_norm, round='nearest',
                            saturate_normals=False, allow_denorm=True,
                            custom_cuda=False):
    """ Core function used for element-wise quantization
    Arguments:
      A         {PyTorch tensor} -- A tensor to be quantized
      bits      {int}            -- Number of mantissa bits. Includes
                                    sign bit and implicit one for floats
      exp_bits  {int}            -- Number of exponent bits, 0 for ints
      max_norm  {float}          -- Largest representable normal number
      round     {str}            -- Rounding mode: (floor, nearest, even)
      saturate_normals {bool}    -- If True, normal numbers (i.e., not NaN/Inf)
                                    that exceed max norm are clamped.
                                    Must be True for correct MX conversion.
      allow_denorm     {bool}    -- If False, flush denorm numbers in the
                                    elem_format to zero.
      custom_cuda      {str}     -- If True, use custom CUDA kernels
    Returns:
      quantized tensor {PyTorch tensor} -- A tensor that has been quantized
    """
    A_is_sparse = A.is_sparse
    if A_is_sparse:
        if A.layout != torch.sparse_coo:
            raise NotImplementedError("Only COO layout sparse tensors are currently supported.")

        sparse_A = A.coalesce()
        A = sparse_A.values().clone()

    # custom cuda only support floor and nearest rounding modes
    custom_cuda = custom_cuda and round in RoundingMode.string_enums()

    if custom_cuda:
        A = A.contiguous()

        from . import custom_extensions
        if A.device.type == "cuda":
            A = custom_extensions.funcs.quantize_elemwise_func_cuda(
                A, bits, exp_bits, max_norm, RoundingMode[round],
                saturate_normals, allow_denorm)
        elif A.device.type == "cpu":
            A = custom_extensions.funcs.quantize_elemwise_func_cpp(
                A, bits, exp_bits, max_norm, RoundingMode[round],
                saturate_normals, allow_denorm)
        return A

    # Flush values < min_norm to zero if denorms are not allowed
    if not allow_denorm and exp_bits > 0:
        min_norm = _get_min_norm(exp_bits)
        out = (torch.abs(A) >= min_norm).type(A.dtype) * A
    else:
        out = A

    if exp_bits != 0:
        private_exp = torch.floor(torch.log2(
            torch.abs(A) + (A == 0).type(A.dtype)))

        # The minimum representable exponent for 8 exp bits is -126
        min_exp = -(2**(exp_bits-1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None

    # Scale up so appropriate number of bits are in the integer portion of the number
    out = _safe_lshift(out, bits - 2, private_exp)

    out = _round_mantissa(out, bits, round, clamp=False)

    # Undo scaling
    out = _safe_rshift(out, bits - 2, private_exp)

    # Set values > max_norm to Inf if desired, else clamp them
    if saturate_normals or exp_bits == 0:
        out = torch.clamp(out, min=-max_norm, max=max_norm)
    else:
        out = torch.where((torch.abs(out) > max_norm),
                           torch.sign(out) * float("Inf"), out)

    # handle Inf/NaN
    if not custom_cuda:
        out[A == float("Inf")] = float("Inf")
        out[A == -float("Inf")] = -float("Inf")
        out[A == float("NaN")] = float("NaN")

    if A_is_sparse:
        output = torch.sparse_coo_tensor(sparse_A.indices(), output,
                sparse_A.size(), dtype=sparse_A.dtype, device=sparse_A.device,
                requires_grad=sparse_A.requires_grad)

    return out


def _quantize_elemwise(A, elem_format, round='nearest', custom_cuda=False,
                       saturate_normals=False, allow_denorm=True):
    """ Quantize values to a defined format. See _quantize_elemwise_core()
    """
    if elem_format == None:
        return A

    ebits, mbits, _, max_norm, _ = _get_format_params(elem_format)

    output = _quantize_elemwise_core(
            A, mbits, ebits, max_norm,
            round=round, allow_denorm=allow_denorm,
            saturate_normals=saturate_normals,
            custom_cuda=custom_cuda)

    return output


def _quantize_bfloat(A, bfloat, round='nearest', custom_cuda=False, allow_denorm=True):
    """ Quantize values to bfloatX format
    Arguments:
      bfloat      {int}       -- Total number of bits for bfloatX format,
                                 Includes 1 sign, 8 exp bits, and variable
                                 mantissa bits. Must be >= 9.
    """
    # Shortcut for no quantization
    if bfloat == 0 or bfloat == 32:
        return A

    max_norm = _get_max_norm(8, bfloat-7)

    return _quantize_elemwise_core(
            A, bits=bfloat-7, exp_bits=8, max_norm=max_norm, round=round,
            allow_denorm=allow_denorm, custom_cuda=custom_cuda)


def _quantize_fp(A, exp_bits=None, mantissa_bits=None,
                 round='nearest', custom_cuda=False, allow_denorm=True):
    """ Quantize values to IEEE fpX format. The format defines NaN/Inf
        and subnorm numbers in the same way as FP32 and FP16.
    Arguments:
        exp_bits        {int} -- number of bits used to store exponent
        mantissa_bits   {int} -- number of bits used to store mantissa, not
                                 including sign or implicit 1
        round           {str} -- Rounding mode, (floor, nearest, even)
    """
    # Shortcut for no quantization
    if exp_bits is None or mantissa_bits is None:
        return A

    max_norm = _get_max_norm(exp_bits, mantissa_bits+2)

    output = _quantize_elemwise_core(
            A, bits=mantissa_bits + 2, exp_bits=exp_bits,
            max_norm=max_norm, round=round, allow_denorm=allow_denorm,
            custom_cuda=custom_cuda)

    return output


def quantize_elemwise_op(A, mx_specs, round=None):
    """A function used for element-wise quantization with mx_specs
    Arguments:
      A          {PyTorch tensor} -- a tensor that needs to be quantized
      mx_specs {dictionary}     -- dictionary to specify mx_specs
      round      {str}            -- Rounding mode, choose from (floor, nearest, even)
                                     (default: "nearest")
    Returns:
      quantized value {PyTorch tensor} -- a tensor that has been quantized
    """
    if mx_specs is None:
        return A
    elif round is None:
        round = mx_specs['round']

    if mx_specs['bfloat'] == 16 and round == 'even'\
        and torch.cuda.is_bf16_supported() \
        and mx_specs['bfloat_subnorms'] == True:
        return A.to(torch.bfloat16)

    if mx_specs['bfloat'] > 0 and mx_specs['fp'] > 0:
        raise ValueError("Cannot set both [bfloat] and [fp] in mx_specs.")
    elif mx_specs['bfloat'] > 9:
        A = _quantize_bfloat(A, bfloat=mx_specs['bfloat'], round=round,
                             custom_cuda=mx_specs['custom_cuda'],
                             allow_denorm=mx_specs['bfloat_subnorms'])
    elif mx_specs['bfloat'] > 0 and mx_specs['bfloat'] <= 9:
        raise ValueError("Cannot set [bfloat] <= 9 in mx_specs.")
    elif mx_specs['fp'] > 6:
        A = _quantize_fp(A, exp_bits=5, mantissa_bits=mx_specs['fp'] - 6,
                         round=round, custom_cuda=mx_specs['custom_cuda'],
                         allow_denorm=mx_specs['bfloat_subnorms'])
    elif mx_specs['fp'] > 0 and mx_specs['fp'] <= 6:
        raise ValueError("Cannot set [fp] <= 6 in mx_specs.")
    return A

