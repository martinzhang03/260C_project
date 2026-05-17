"""Wanda-style unstructured pruning for dense HF modules (nn.Linear / Conv1D).

Fix history
-----------
The original code thresholded a single ``kthvalue`` over the *whole*
flattened score matrix. The Wanda paper instead compares weights **inside
each output neuron's comparison group** (per output row), which keeps every
output channel alive and is markedly more robust at 30-50% sparsity. We now
prune per output channel by default; the global variant is kept only for the
ablation table.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from . import hf_layers
from .hf_layers import is_weight_matrix_module


def wanda_scores_linear(weight_out_in: torch.Tensor, act_rms: torch.Tensor) -> torch.Tensor:
    """Linear weight [out,in], act_rms [in]."""
    if act_rms.device != weight_out_in.device:
        act_rms = act_rms.to(weight_out_in.device)
    return weight_out_in.abs() * act_rms.view(1, -1)


def wanda_scores_conv1d(weight_nx_nf: torch.Tensor, act_rms: torch.Tensor) -> torch.Tensor:
    """GPT-2 Conv1D stores weight [nx,nf]; act RMS is length nx."""
    if act_rms.device != weight_nx_nf.device:
        act_rms = act_rms.to(weight_nx_nf.device)
    return weight_nx_nf.abs() * act_rms.view(-1, 1)


def _mask_per_output(scores: torch.Tensor, out_axis: int, sparsity: float) -> torch.Tensor:
    """Boolean keep-mask, pruning ``sparsity`` of each output channel's group.

    Wanda's comparison group = the incoming weights of one output neuron.
    """
    in_axis = 1 - out_axis
    n_in = scores.shape[in_axis]
    k = int(round(n_in * sparsity))
    if k <= 0:
        return torch.ones_like(scores, dtype=torch.bool)
    # Per-row (per output channel) k-th smallest score.
    thresh = torch.kthvalue(scores, k, dim=in_axis, keepdim=True).values
    return scores > thresh


def _prune_weight_tensor(
    scores: torch.Tensor,
    tensor: torch.Tensor,
    sparsity: float,
    out_axis: int,
    return_mask: bool,
    *,
    per_output: bool = True,
):
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1).")
    if sparsity == 0.0:
        return torch.ones_like(tensor, dtype=torch.bool) if return_mask else None

    if per_output:
        mask = _mask_per_output(scores, out_axis, sparsity)
    else:
        flat = scores.flatten()
        k = int(round(flat.numel() * sparsity))
        thresh = torch.kthvalue(flat, k).values
        mask = scores > thresh

    tensor *= mask.to(tensor.dtype)
    return mask if return_mask else None


def prune_dense_module(
    module: nn.Module,
    act_rms: torch.Tensor,
    sparsity: float,
    *,
    return_mask: bool = False,
    per_output: bool = True,
) -> Optional[torch.Tensor]:
    if isinstance(module, nn.Linear):
        w = module.weight.data
        scores = wanda_scores_linear(w, act_rms)
        return _prune_weight_tensor(scores, w, sparsity, 0, return_mask, per_output=per_output)
    if hf_layers.Conv1D is not None and isinstance(module, hf_layers.Conv1D):
        w = module.weight.data
        scores = wanda_scores_conv1d(w, act_rms)
        return _prune_weight_tensor(scores, w, sparsity, 1, return_mask, per_output=per_output)

    raise TypeError(module)


def apply_wanda_to_model(
    model: nn.Module,
    act_norms: dict[str, torch.Tensor],
    sparsity: float,
    *,
    per_output: bool = True,
) -> None:
    for fq, module in model.named_modules():
        if is_weight_matrix_module(module) and fq in act_norms:
            prune_dense_module(module, act_norms[fq], sparsity, per_output=per_output)
