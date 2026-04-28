"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import argparse
import collections
import json
import os
import traceback

import torch

_ASSERT_MODE = os.environ.get("MX_ASSERT", "False")


class MxSpecs(collections.UserDict):
    """Container for MX quantization options."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        defaults = {
            "scale_bits": 0,
            "w_elem_format": None,
            "a_elem_format": None,
            "w_elem_format_bp": None,
            "a_elem_format_bp": None,
            "a_elem_format_bp_ex": None,
            "a_elem_format_bp_os": None,
            "mx_flush_fp32_subnorms": False,
            "shared_exp_method": "max",
            "block_size": 0,
            "bfloat": 0,
            "fp": 0,
            "bfloat_subnorms": True,
            "quantize_backprop": True,
            "round": "nearest",
            "round_m": "nearest",
            "round_weight": "nearest",
            "round_output": "nearest",
            "round_grad_weight": "nearest",
            "round_grad_input": "nearest",
            "round_mx_output": "nearest",
            "round_mx_input_grad_input": "nearest",
            "round_mx_weight_grad_input": "nearest",
            "round_mx_grad_output_grad_input": "nearest",
            "round_mx_input_grad_weight": "nearest",
            "round_mx_grad_output_grad_weight": "nearest",
            "softmax_exp2": False,
            "vec_use_exp2": False,
            "vec_use_recip": False,
            "custom_cuda": False,
            "lns_mode": False,
            "prequantized": False,
        }

        self.help_strings = {
            "scale_bits": "Bits used for the shared exponent or scale.",
            "w_elem_format": "Weight MX element format.",
            "a_elem_format": "Activation MX element format.",
            "w_elem_format_bp": "Backward-pass weight MX element format.",
            "a_elem_format_bp": "Backward-pass activation MX element format.",
            "a_elem_format_bp_ex": "Backward-pass activation grad MX element format.",
            "a_elem_format_bp_os": "Backward-pass stashed activation MX element format.",
            "mx_flush_fp32_subnorms": "Flush blocks with subnormal shared scales to zero.",
            "shared_exp_method": "Shared exponent calculation method.",
            "block_size": "MX shared exponent block size.",
            "bfloat": "BFloat format width.",
            "fp": "Floating-point format width.",
            "bfloat_subnorms": "Whether BFloat or FP supports subnormals.",
            "quantize_backprop": "Enable MX or BFloat quantization on backward pass.",
            "round": "Global rounding mode.",
            "round_m": "Optimizer state rounding mode.",
            "round_weight": "Weight rounding mode.",
            "round_output": "Activation rounding mode.",
            "round_grad_weight": "Weight gradient rounding mode.",
            "round_grad_input": "Input gradient rounding mode.",
            "round_mx_output": "Forward MX rounding mode.",
            "round_mx_input_grad_input": "",
            "round_mx_weight_grad_input": "",
            "round_mx_grad_output_grad_input": "",
            "round_mx_input_grad_weight": "",
            "round_mx_grad_output_grad_weight": "",
            "softmax_exp2": "Use 2^x in softmax.",
            "vec_use_exp2": "Use 2^x when approximating e^x.",
            "vec_use_recip": "Use reciprocal when approximating division.",
            "custom_cuda": "Enable custom CUDA kernels for quantization.",
            "lns_mode": "Enable LNS quantization mode.",
            "prequantized": "Treat weights as already quantized.",
        }

        for key, value in defaults.items():
            if key not in self.data:
                self.data[key] = value

        for key in self.data:
            assert key in self.help_strings

    def safe_json(self, indent=None):
        default = lambda obj: f"<<non-serializable: {type(obj).__qualname__}>>"
        return json.dumps(self.data, indent=indent, default=default)

    def __str__(self):
        return self.safe_json(indent=4)


def get_default_mx_specs():
    return MxSpecs()


def get_backwards_mx_specs(specs):
    """Return a no-quantization spec when backprop quantization is disabled."""
    bspecs = specs.copy()

    if bspecs["quantize_backprop"] is False:
        bspecs["w_elem_format"] = None
        bspecs["a_elem_format"] = None
        bspecs["w_elem_format_bp"] = None
        bspecs["a_elem_format_bp"] = None
        bspecs["a_elem_format_bp_os"] = None
        bspecs["a_elem_format_bp_ex"] = None
        bspecs["block_size"] = 0
        bspecs["bfloat"] = 0
        bspecs["fp"] = 0

    return bspecs


def apply_mx_specs(mx_specs, default_mx_specs=None):
    """Merge a partial spec dict with defaults."""
    if not default_mx_specs:
        default_mx_specs = get_default_mx_specs()

    if not mx_specs:
        return default_mx_specs

    for key, value in mx_specs.items():
        if value is not None:
            if key not in default_mx_specs:
                raise KeyError(f"Unknown key '{key}' passed to mx specs")
            default_mx_specs[key] = value

    return default_mx_specs


def add_mx_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("mx", "MX specs")
    group.add_argument("--mx_dir", type=str, default=None, help="Path to mx library")

    default_specs = get_default_mx_specs()
    for key, value in default_specs.items():
        help_str = default_specs.help_strings[key] or "No help string"

        if "elem_format" in key:
            group.add_argument(f"--{key}", type=str, default=value, help=help_str)
        elif isinstance(value, bool) and value is False:
            group.add_argument(f"--{key}", action="store_true", help=help_str)
        elif isinstance(value, bool) and value is True:
            group.add_argument(f"--no_{key}", action="store_true", help=help_str)
        else:
            group.add_argument(f"--{key}", type=type(value), default=None, help=help_str)

    group.add_argument(
        "--skip_early_exit",
        action="store_true",
        default=False,
        help="Do not early-exit when quantization is disabled.",
    )
    return parser


def finalize_mx_specs(specs, early_exit=True):
    """Resolve dependent fields and return a fully-populated spec object."""
    if (
        not specs.get("w_elem_format", 0)
        and not specs.get("a_elem_format", 0)
        and not specs.get("w_elem_format_bp", 0)
        and not specs.get("a_elem_format_bp", 0)
        and not specs.get("a_elem_format_bp_os", 0)
        and not specs.get("a_elem_format_bp_ex", 0)
        and not specs.get("bfloat", 0)
        and not specs.get("fp", 0)
        and early_exit
    ):
        return None

    if specs.get("custom_cuda"):
        assert torch.cuda.is_available(), "'custom_cuda' is only supported on CUDA devices."

    def assign_if_none(target, source):
        if (target not in specs or specs[target] is None) and source in specs:
            specs[target] = specs[source]

    assign_if_none("w_elem_format_bp", "w_elem_format")
    assign_if_none("a_elem_format_bp", "a_elem_format")
    assign_if_none("a_elem_format_bp_os", "a_elem_format")
    assign_if_none("a_elem_format_bp_ex", "a_elem_format")

    assign_if_none("round_m", "round")
    assign_if_none("round_output", "round")
    assign_if_none("round_grad_weight", "round")
    assign_if_none("round_grad_input", "round")
    assign_if_none("round_weight", "round")
    assign_if_none("round_mx_output", "round")

    assign_if_none("round_mx_input_grad_input", "round_grad_input")
    assign_if_none("round_mx_weight_grad_input", "round_grad_input")
    assign_if_none("round_mx_grad_output_grad_input", "round_grad_input")
    assign_if_none("round_mx_input_grad_weight", "round_grad_input")
    assign_if_none("round_mx_grad_output_grad_weight", "round_grad_input")

    return apply_mx_specs(specs, get_default_mx_specs())


def get_mx_specs(parsed_args: argparse.Namespace):
    default_specs = get_default_mx_specs()
    parsed_specs = {}

    for key, value in default_specs.items():
        if isinstance(value, bool) and value is True:
            arg_key = "no_" + key
            if hasattr(parsed_args, arg_key):
                parsed_specs[key] = not getattr(parsed_args, arg_key)
        elif hasattr(parsed_args, key):
            parsed_specs[key] = getattr(parsed_args, key)

    early_exit = not getattr(parsed_args, "skip_early_exit", False)
    return finalize_mx_specs(parsed_specs, early_exit=early_exit)


def mx_assert_test(mx_specs):
    if _ASSERT_MODE == "True" and mx_specs is None:
        stack = traceback.extract_stack()
        failing_func = stack[-2]
        call_site = stack[-3]
        raise ValueError(
            "MX assert test failed!\n"
            f"mx_specs is None in function {failing_func.name}\n"
            f"Called from {call_site.filename}, line {call_site.lineno}\n"
            f"  {call_site.line}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_mx_args(parser)

    args = parser.parse_args([])
    specs = get_mx_specs(args)
    assert specs is None

    args = parser.parse_args([])
    args.bfloat = 4
    specs = get_mx_specs(args)
    assert specs["bfloat"] == 4

    defaults = get_default_mx_specs()
    for key, value in specs.items():
        if key != "bfloat":
            assert defaults[key] == value, (key, defaults[key], value)
    for key, value in defaults.items():
        if key != "bfloat":
            assert specs[key] == value, (key, value, specs[key])

    print("Passed!")
