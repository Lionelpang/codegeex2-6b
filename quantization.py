from torch.nn import Linear
from torch.nn.parameter import Parameter

import bz2
import torch
import base64
import ctypes
from transformers.utils import logging

from typing import List
from functools import partial

logger = logging.get_logger(__name__)

try:
    from cpm_kernels.kernels.base import LazyKernelCModule, KernelFunction, round_up

    class Kernel:
        def __init__(self, code: bytes, function_names: List[str]):
            self.code = code
            self._function_names = function_names
            self._cmodule = LazyKernelCModule(self.code)

            for name in self._function_names:
                setattr(self, name, KernelFunction(self._cmodule, name))

    quantization_code = "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"

    kernels = Kernel(
        bz2.decompress(base64.b64decode(quantization_code)),
        [
            "int4WeightCompression",
            "int4WeightExtractionFloat",
            "int4WeightExtractionHalf",
            "int4WeightExtractionBFloat16",
            "int8WeightExtractionFloat",
            "int8WeightExtractionHalf",
            "int8WeightExtractionBFloat16",
        ],
    )
except Exception as exception:
    kernels = None
    logger.warning("Failed to load cpm_kernels:" + str(exception))


class W8A16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, quant_w: torch.Tensor, scale_w: torch.Tensor, weight_bit_width):
        ctx.inp_shape = inp.size()
        ctx.weight_bit_width = weight_bit_width
        out_features = quant_w.size(0)
        inp = inp.contiguous().view(-1, inp.size(-1))
        weight = extract_weight_to_half(quant_w, scale_w, weight_bit_width)
        ctx.weight_shape = weight.size()
        output = inp.mm(weight.t())
        ctx.save_for_backward(inp, quant_w, scale_w)
        return output.view(*(ctx.inp_shape[:-1] + (out_features,)))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inp, quant_w, scale_w = ctx.saved_tensors
        weight = extract_weight_to_half(quant_w, scale_w, ctx.weight_bit_width)
        grad_output = grad_output.contiguous().view(-1, weight.size(0))
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(inp)
        return grad_input.view(ctx.inp_shape), grad_weight.view(ctx.weight_shape), None, None


def compress_int4_weight(weight: torch.Tensor):  # (n, m)
    with torch.cuda.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        assert m % 2 == 0
        m = m // 2
        out = torch.empty(n, m, dtype=torch.int8, device="cuda")
        stream = torch.cuda.current_stream()

        gridDim = (n, 1, 1)
        blockDim = (min(round_up(m, 32), 1024), 1, 1)

        kernels.int4WeightCompression(
            gridDim,
            blockDim,
            0,
            stream,
            [ctypes.c_void_p(weight.data_ptr()), ctypes.c_void_p(out.data_ptr()), ctypes.c_int32(n), ctypes.c_int32(m)],
        )
        return out


def extract_weight_to_half(weight: torch.Tensor, scale_list: torch.Tensor, source_bit_width: int):
    assert scale_list.dtype in [torch.half, torch.bfloat16]
    assert weight.dtype in [torch.int8]
    if source_bit_width == 8:
        return weight.to(scale_list.dtype) * scale_list[:, None]
    elif source_bit_width == 4:
        func = (
            kernels.int4WeightExtractionHalf if scale_list.dtype == torch.half else kernels.int4WeightExtractionBFloat16
        )
    else:
        assert False, "Unsupported bit-width"

    with torch.cuda.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        out = torch.empty(n, m * (8 // source_bit_width), dtype=scale_list.dtype, device="cuda")
        stream = torch.cuda.current_stream()

        gridDim = (n, 1, 1)
        blockDim = (min(round_up(m, 32), 1024), 1, 1)

        func(
            gridDim,
            blockDim,
            0,
            stream,
            [
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(scale_list.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int32(n),
                ctypes.c_int32(m),
            ],
        )
        return out


class QuantizedLinear(torch.nn.Module):
    def __init__(self, weight_bit_width: int, weight, bias=None, device="cpu", dtype=None, empty_init=False, *args,
                 **kwargs):
        super().__init__()
        self.weight_bit_width = weight_bit_width

        shape = weight.shape

        if weight is None or empty_init:
            self.weight = torch.empty(shape[0], shape[1] * weight_bit_width // 8, dtype=torch.int8, device=device)
            self.weight_scale = torch.empty(shape[0], dtype=dtype, device=device)
        else:
            self.weight_scale = weight.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)
            self.weight = torch.round(weight / self.weight_scale[:, None]).to(torch.int8)
            if weight_bit_width == 4:
                self.weight = compress_int4_weight(self.weight)

        self.weight = Parameter(self.weight.to(device), requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale.to(device), requires_grad=False)
        self.bias = Parameter(bias.to(device), requires_grad=False) if bias is not None else None

    def forward(self, input):
        output = W8A16Linear.apply(input, self.weight, self.weight_scale, self.weight_bit_width)
        if self.bias is not None:
            output = output + self.bias
        return output


def quantize(model, weight_bit_width, empty_init=False, device=None):
    """Replace fp16 linear with quantized linear"""
    for layer in model.layers:
        layer.self_attention.query_key_value = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.self_attention.query_key_value.weight.to(torch.cuda.current_device()),
            bias=layer.self_attention.query_key_value.bias,
            dtype=layer.self_attention.query_key_value.weight.dtype,
            device=layer.self_attention.query_key_value.weight.device if device is None else device,
            empty_init=empty_init
        )
        layer.self_attention.dense = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.self_attention.dense.weight.to(torch.cuda.current_device()),
            bias=layer.self_attention.dense.bias,
            dtype=layer.self_attention.dense.weight.dtype,
            device=layer.self_attention.dense.weight.device if device is None else device,
            empty_init=empty_init
        )
        layer.mlp.dense_h_to_4h = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.mlp.dense_h_to_4h.weight.to(torch.cuda.current_device()),
            bias=layer.mlp.dense_h_to_4h.bias,
            dtype=layer.mlp.dense_h_to_4h.weight.dtype,
            device=layer.mlp.dense_h_to_4h.weight.device if device is None else device,
            empty_init=empty_init
        )
        layer.mlp.dense_4h_to_h = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.mlp.dense_4h_to_h.weight.to(torch.cuda.current_device()),
            bias=layer.mlp.dense_4h_to_h.bias,
            dtype=layer.mlp.dense_4h_to_h.weight.dtype,
            device=layer.mlp.dense_4h_to_h.weight.device if device is None else device,
            empty_init=empty_init
        )

    return model
