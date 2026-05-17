"""Detect layer types used by common Hugging Face causal LMs (Linear + GPT-2 Conv1D)."""

from __future__ import annotations

import torch.nn as nn

_WEIGHT_TYPES: tuple[type, ...] = (nn.Linear,)
Conv1D: type | None
try:
    from transformers.pytorch_utils import Conv1D as _HFConv1D

    Conv1D = _HFConv1D
    _WEIGHT_TYPES = (nn.Linear, _HFConv1D)
except Exception:  # pragma: no cover
    Conv1D = None


def is_weight_matrix_module(m: nn.Module) -> bool:
    return isinstance(m, _WEIGHT_TYPES)


def weight_and_in_dim(m: nn.Module) -> tuple[nn.Parameter, int]:
    if isinstance(m, nn.Linear):
        return m.weight, m.in_features
    if Conv1D is not None and isinstance(m, Conv1D):
        return m.weight, m.nx
    raise TypeError(f"Unsupported module {type(m)}")
