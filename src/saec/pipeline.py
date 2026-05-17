"""Stage 1 — activation-aware pruning pipeline.

End-to-end: calibration → Wanda-style per-output pruning, with the token
embedding and (weight-tied) output head kept dense. Mixed-precision
quantization and structure-aware FP16 routing arrive in Stage 2.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
import torch.nn as nn

from .calibration import collect_activation_norms_for_linears
from .hf_layers import is_weight_matrix_module
from .structure import LayerKind, classify_linear_name
from .wanda import apply_wanda_to_model


class CompressionPipelineConfig:
    def __init__(
        self,
        *,
        prune_sparsity: float = 0.5,
        per_output_prune: bool = True,
    ):
        self.prune_sparsity = prune_sparsity
        self.per_output_prune = per_output_prune


def run_calibration(
    model: nn.Module,
    batches: Iterator[dict[str, torch.Tensor]],
    device: torch.device,
    max_batches: int,
    show_progress: bool = True,
) -> dict[str, torch.Tensor]:
    return collect_activation_norms_for_linears(
        model,
        batches,
        device=device,
        max_batches=max_batches,
        show_progress=show_progress,
    )


def protected_modules(model: nn.Module) -> set[str]:
    """Layers never pruned.

    The token embedding and the (in GPT-2, weight-tied) output head are kept
    dense — pruning the tied matrix corrupts the input embedding and collapses
    the model. Skipping embedding/head is standard in Wanda, GPTQ and AWQ.
    """
    out: set[str] = set()
    for n, m in model.named_modules():
        if is_weight_matrix_module(m) and classify_linear_name(n) in (
            LayerKind.LM_HEAD,
            LayerKind.EMBEDDING,
        ):
            out.add(n)
    return out


def compress_inplace(
    model: nn.Module,
    act_norms: dict[str, torch.Tensor],
    cfg: CompressionPipelineConfig,
    *,
    device: Optional[torch.device] = None,
) -> dict[str, torch.Tensor]:
    if device is not None:
        model = model.to(device)
    protected = protected_modules(model)
    if cfg.prune_sparsity > 0.0:
        prune_norms = {k: v for k, v in act_norms.items() if k not in protected}
        apply_wanda_to_model(
            model, prune_norms, cfg.prune_sparsity, per_output=cfg.per_output_prune
        )
    return act_norms
