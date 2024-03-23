import warnings
from numbers import Number
from typing import Any, Callable, List, Optional, Union
import numpy as np
from fvcore.nn.jit_handles import get_shape
from fvcore.nn.flop_count import FlopCountAnalysis, _DEFAULT_SUPPORTED_OPS, flop_count

try:
    from math import prod
except ImportError:
    from numpy import prod

warnings.filterwarnings("ignore")


def leaky_relu_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    out_shape = get_shape(outputs[0])
    flops = prod(out_shape)
    return flops


def mul_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    inp1_shape = get_shape(inputs[0])
    flops = prod(inp1_shape)
    return flops


def add_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    inp1_shape = get_shape(inputs[0])
    flops = prod(inp1_shape)
    return flops


def square_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    inp1_shape = get_shape(inputs[0])
    flops = prod(inp1_shape)
    return flops


def mean_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    inp1_shape = get_shape(inputs[0])
    out_shape = get_shape(outputs[0])
    return prod(inp1_shape)


def rsqrt_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    inp1_shape = get_shape(inputs[0])
    flops = prod(inp1_shape)
    return flops


def sum_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    inp1_shape = get_shape(inputs[0])
    flops = prod(inp1_shape)
    return flops


def pointwise_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    inp1_shape = get_shape(inputs[0])
    flops = prod(inp1_shape)
    return flops


def rfft_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the rfft/rfftn operator.
    Source: https://github.com/raoyongming/DynamicViT/blob/master/calc_flops.py
    """
    input_shape = inputs[0].type().sizes()
    B, H, W, C = input_shape
    N = H * W
    flops = N * C * np.ceil(np.log2(N))
    return flops


supported_ops = {
    **_DEFAULT_SUPPORTED_OPS,
    "aten::leaky_relu_": leaky_relu_jit,
    "aten::mul": mul_jit,
    "aten::add": add_jit,
    "aten::sub": add_jit,
    "aten::add_": add_jit,
    "aten::square": square_jit,
    "aten::mean": mean_jit,
    "aten::rsqrt": rsqrt_jit,
    "aten::sum": sum_jit,
    "aten::fft_rfftn": rfft_flop_jit,
    "aten::real": pointwise_jit,
    "aten::imag": pointwise_jit,
    "aten::complex": pointwise_jit,
    "aten::fft_irfftn": rfft_flop_jit,
    "aten::sigmoid": leaky_relu_jit
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_flops(model, inputs):
    flops = FlopCountAnalysis(model, (*inputs, )).set_op_handle(**supported_ops)
    flops_ = flops.total() / 1e9
    param = count_parameters(model) / 1e6
    print(f"FLOPs: {flops_:.4f} G")
    print(f"Total number of parameters : {param:.5f} M")
    return flops_, param

