"""Calibration utilities: collect activation-driven statistics used by Wanda-style pruning."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .hf_layers import is_weight_matrix_module, weight_and_in_dim


def _rms_per_input_dim(hidden_states: torch.Tensor) -> torch.Tensor:
    flat = hidden_states.detach().flatten(0, -2).float()
    if flat.numel() == 0:
        return torch.zeros(hidden_states.shape[-1], device=hidden_states.device)
    sq = torch.sum(flat**2, dim=0)
    return torch.sqrt(sq / max(flat.shape[0], 1))


def collect_activation_norms_for_linears(
    model: nn.Module,
    calibration_batches: Iterator[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    max_batches: Optional[int] = None,
    show_progress: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Accumulate RMS norms of incoming activations for each dense weight module (Linear / HF Conv1D).

    Returned dict maps module fq name -> RMS vector of shape [in_features/nx].
    """
    _ = dtype
    model = model.to(device)
    sums: defaultdict[str, torch.Tensor] = defaultdict()
    counts: defaultdict[str, int] = defaultdict(int)

    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str, in_features: int):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]):
            nonlocal sums, counts
            x = inputs[0].detach().to(torch.float32)
            rms = _rms_per_input_dim(x)
            if rms.numel() != in_features:
                raise ValueError(f"Unexpected activation dim for {name}: {rms.numel()} vs {in_features}")
            if name not in sums:
                sums[name] = torch.zeros_like(rms, device="cpu")
            sums[name] += rms.cpu()
            counts[name] += 1

        return hook

    for fq, module in model.named_modules():
        if is_weight_matrix_module(module):
            _w, inn = weight_and_in_dim(module)
            _ = _w  # symmetry with future extensions
            handles.append(module.register_forward_pre_hook(make_hook(fq, inn)))

    it = calibration_batches
    if show_progress and max_batches is not None:
        it = tqdm(it, total=max_batches, desc="calibration")
    elif show_progress:
        it = tqdm(it, desc="calibration")

    batch_count = 0
    model.eval()
    with torch.no_grad():
        for batch in it:
            if max_batches is not None and batch_count >= max_batches:
                break
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
                model(**batch)
            batch_count += 1

    for h in handles:
        h.remove()

    out: dict[str, torch.Tensor] = {}
    for name, s in sums.items():
        c = max(counts[name], 1)
        out[name] = (s / c).contiguous()
    return out
