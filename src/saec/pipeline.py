"""End-to-end: calibration → prune (Wanda) → structure-aware mixed precision.

Routing modes for the FP16 keep-set (the "structure-aware" decision):

* ``salience``  – protect the layers whose INT{N} quantization perturbs the
  output the most, estimated as Σ act_rms² · (ΔW)² (activation-weighted
  quantization error). This is the principled structure-aware criterion and
  the default.
* ``kind``      – heuristic: protect lm_head / attention projections first
  (the original behaviour; kept for the routing ablation).
* ``uniform``   – protect a fixed arbitrary slice (name-sorted). A
  structure-blind control.
* ``random``    – protect a random subset (seeded). The other control.
"""

from __future__ import annotations

import random
from typing import Iterator, Optional

import torch
import torch.nn as nn

from .calibration import collect_activation_norms_for_linears
from .hf_layers import is_weight_matrix_module
from .mixed_precision_quant import apply_mixed_precision_modules, fake_quant_symmetric_intn
from .structure import LayerKind, classify_linear_name, structural_fp16_candidates, tag_dense_weight_modules
from .wanda import apply_wanda_to_model


class CompressionPipelineConfig:
    def __init__(
        self,
        *,
        prune_sparsity: float = 0.5,
        low_bitwidth: int = 3,
        fp16_fraction: float = 0.05,
        prefer_attention_fp16: bool = True,
        routing: str = "salience",
        routing_seed: int = 0,
        per_channel_quant: bool = True,
        per_output_prune: bool = True,
        group_size: int = 0,
    ):
        self.prune_sparsity = prune_sparsity
        self.low_bitwidth = low_bitwidth
        self.fp16_fraction = fp16_fraction
        self.prefer_attention_fp16 = prefer_attention_fp16
        self.routing = routing
        self.routing_seed = routing_seed
        self.per_channel_quant = per_channel_quant
        self.per_output_prune = per_output_prune
        self.group_size = group_size


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


def _quant_sensitivity(
    module: nn.Module, act_rms: Optional[torch.Tensor], bits: int, group_size: int = 0
) -> float:
    """Activation-weighted INT{bits} quantization error of one dense layer.

    Estimates E‖Δy‖² = Σ_in act_rms[in]² · ‖ΔW[:, in]‖²  (Linear);
    the Conv1D layout is transposed. Higher → more damaged by quantization
    → better FP16-protection candidate.
    """
    w = module.weight.data.float()
    if isinstance(module, nn.Linear):
        out_axis, in_axis = 0, 1
    else:  # Conv1D [nx(in), nf(out)]
        out_axis, in_axis = 1, 0
    dq, _, _ = fake_quant_symmetric_intn(
        w, bits, per_channel=True, out_axis=out_axis, group_size=group_size
    )
    err2 = (w - dq) ** 2  # [out,in] or [in,out]
    per_in = err2.sum(dim=out_axis)  # length = in_features
    if act_rms is not None and act_rms.numel() == per_in.numel():
        per_in = per_in * (act_rms.to(per_in.device).float() ** 2)
    return float(per_in.sum().item())


def protected_modules(model: nn.Module) -> set[str]:
    """Layers never pruned or quantized.

    The token embedding and the (in GPT-2, weight-tied) output head are kept
    in full precision — quantizing the tied matrix corrupts the input
    embedding and collapses the model. Skipping embedding/head is standard in
    Wanda, GPTQ and AWQ.
    """
    out: set[str] = set()
    for n, m in model.named_modules():
        if is_weight_matrix_module(m) and classify_linear_name(n) in (
            LayerKind.LM_HEAD,
            LayerKind.EMBEDDING,
        ):
            out.add(n)
    return out


def select_fp16_modules(
    model: nn.Module,
    act_norms: dict[str, torch.Tensor],
    cfg: CompressionPipelineConfig,
) -> set[str]:
    protected = protected_modules(model)
    dense = [
        n
        for n, m in model.named_modules()
        if is_weight_matrix_module(m) and n not in protected
    ]
    total = len(dense)
    if total == 0:
        return set(protected)
    if cfg.fp16_fraction <= 0.0:
        return set(protected)
    k = max(0, min(total, int(round(total * cfg.fp16_fraction))))
    if k == 0:
        return set(protected)

    mode = cfg.routing
    dense_set = set(dense)
    if mode == "kind":
        tags = tag_dense_weight_modules(model)
        cand = [n for n in structural_fp16_candidates(tags, cfg.prefer_attention_fp16) if n in dense_set]
        sel = cand[:k]
    elif mode == "uniform":
        sel = sorted(dense)[:k]
    elif mode == "random":
        rng = random.Random(cfg.routing_seed)
        sel = rng.sample(dense, k)
    elif mode == "salience":
        mods = dict(model.named_modules())
        sel = sorted(
            dense,
            key=lambda n: _quant_sensitivity(
                mods[n], act_norms.get(n), cfg.low_bitwidth, cfg.group_size
            ),
            reverse=True,
        )[:k]
    else:
        raise ValueError(f"unknown routing mode {mode!r}")
    return set(sel) | protected


# Back-compat alias used by older callers / tests.
def select_structure_aware_fp16_modules(model: nn.Module, cfg: CompressionPipelineConfig) -> set[str]:
    return select_fp16_modules(model, {}, cfg)


def compress_inplace(
    model: nn.Module,
    act_norms: dict[str, torch.Tensor],
    cfg: CompressionPipelineConfig,
    *,
    device: Optional[torch.device] = None,
    do_prune: bool = True,
    do_quant: bool = True,
) -> tuple[set[str], dict[str, torch.Tensor]]:
    if device is not None:
        model = model.to(device)
    protected = protected_modules(model)
    fp16_keep = select_fp16_modules(model, act_norms, cfg) if do_quant else set(protected)
    if do_prune and cfg.prune_sparsity > 0.0:
        prune_norms = {k: v for k, v in act_norms.items() if k not in protected}
        apply_wanda_to_model(
            model, prune_norms, cfg.prune_sparsity, per_output=cfg.per_output_prune
        )
    if do_quant:
        apply_mixed_precision_modules(
            model, fp16_keep, low_bits=cfg.low_bitwidth,
            per_channel=cfg.per_channel_quant, group_size=cfg.group_size,
        )
    return fp16_keep, act_norms
