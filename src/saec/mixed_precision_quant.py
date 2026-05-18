"""Simulated mixed-precision quantization (QDQ on weights only).

Fix history
-----------
The original implementation used a single *per-tensor* symmetric scale
(``scale = max(|W|) / qmax``). On Transformer weight matrices a handful of
outlier weights inflate that scalar so the bulk of the matrix is quantized
with only a few effective levels. Every quantization paper that quotes
"<1% perplexity at INT8" (RTN, GPTQ, AWQ) uses **per-output-channel**
scales. Per-tensor was the dominant cause of the +10-13% perplexity
regression. We now default to per-output-channel symmetric quantization
and keep per-tensor available only for ablations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import hf_layers
from .hf_layers import is_weight_matrix_module


def _out_axis(module: nn.Module) -> int:
    """Dimension of ``module.weight`` that indexes output channels.

    nn.Linear stores ``[out, in]`` -> axis 0.
    GPT-2 Conv1D stores ``[nx(in), nf(out)]`` -> axis 1.
    """
    if isinstance(module, nn.Linear):
        return 0
    if hf_layers.Conv1D is not None and isinstance(module, hf_layers.Conv1D):
        return 1
    raise TypeError(f"Unsupported module {type(module)}")


def fake_quant_symmetric_intn(
    weight: torch.Tensor,
    bits: int,
    *,
    per_channel: bool = True,
    out_axis: int = 0,
    group_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric intN quantization simulated via a QDQ round-trip.

    per_channel=False -> single per-tensor scale (legacy / ablation path).
    per_channel=True, group_size<=0 -> one scale per output channel.
    per_channel=True, group_size=G  -> one scale per (output channel, block
        of G input weights). Group-wise quantization is the standard recipe
        AWQ / GPTQ / llama.cpp use to make INT3/INT4 viable; without it naive
        RTN collapses GPT-2 below 4 bits.
    """
    if bits < 2:
        raise ValueError("bits must be >= 2 for meaningful quantization.")
    qmax = 2 ** (bits - 1) - 1
    w = weight
    eps = torch.finfo(w.dtype).eps

    if not per_channel:
        scale = torch.clamp(torch.max(w.abs()) / float(qmax), min=eps)
        q = torch.clamp(torch.round(w / scale), -qmax, qmax)
        return q * scale, scale, q

    in_axis = 1 - out_axis
    n_in = w.shape[in_axis]
    if group_size and 0 < group_size < n_in and n_in % group_size == 0:
        # View input dim as [n_groups, group_size]; scale per (out, group).
        wt = w if out_axis == 0 else w.transpose(0, 1)  # -> [out, in]
        out_f, in_f = wt.shape
        g = wt.reshape(out_f, in_f // group_size, group_size)
        scale = torch.clamp(g.abs().amax(dim=2, keepdim=True) / float(qmax), min=eps)
        q = torch.clamp(torch.round(g / scale), -qmax, qmax)
        dq = (q * scale).reshape(out_f, in_f)
        if out_axis != 0:
            dq = dq.transpose(0, 1).contiguous()
        return dq, scale, q

    reduce_dims = [d for d in range(w.dim()) if d != out_axis]
    scale = torch.clamp(w.abs().amax(dim=reduce_dims, keepdim=True) / float(qmax), min=eps)
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q * scale, scale, q


def quantize_module_(
    module: nn.Module, bits: int, *, per_channel: bool = True, group_size: int = 0
) -> None:
    """In-place QDQ of one dense weight module."""
    axis = _out_axis(module)
    dq, _, _ = fake_quant_symmetric_intn(
        module.weight.data.float(), bits,
        per_channel=per_channel, out_axis=axis, group_size=group_size,
    )
    module.weight.data = dq.to(module.weight.dtype)


def apply_mixed_precision_modules(
    model: nn.Module,
    fp16_module_names: set[str],
    low_bits: int = 3,
    *,
    per_channel: bool = True,
    group_size: int = 0,
) -> None:
    """QDQ every dense weight tensor except the structure-aware FP16 keep set."""
    for fq, m in model.named_modules():
        if not is_weight_matrix_module(m):
            continue
        if fq in fp16_module_names:
            continue
        quantize_module_(m, low_bits, per_channel=per_channel, group_size=group_size)
