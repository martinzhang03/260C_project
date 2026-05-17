"""Benchmark latency and perplexity for compressed vs baseline causal LMs."""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn


@torch.no_grad()
def latency_ms_per_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    warmup: int = 3,
    reps: int = 20,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    ids = input_ids.to(device)

    def one():
        logits = model(input_ids=ids, use_cache=False).logits
        return logits

    for _ in range(warmup):
        one()

    if device.type == "cuda":
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        one()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(sum(times) / len(times))


@torch.no_grad()
def perplexity_on_batch_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    logits: [B,T,V], labels: [B,T]. Standard next-token shift CE averaged as exp(nll).
    """
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab = shift_logits.shape[-1]
    loss = nn.functional.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="mean",
    )
    return float(torch.exp(loss).item())


@torch.no_grad()
def mean_nll_from_logits(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab = shift_logits.shape[-1]
    return nn.functional.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="mean",
    )


@torch.no_grad()
def perplexity_hf(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    if hasattr(out, "loss") and out.loss is not None:
        return float(torch.exp(out.loss).item())
    logits = out.logits
    return perplexity_on_batch_logits(logits, input_ids)


@torch.no_grad()
def perplexity_hf_batches_mean(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> float:
    """
    Mean causal LM negative log-likelihood across batches → exp(mean NLL).

    Equivalent to perplexity_hf on a single batch when HF emits `loss`; works when `labels` yields logits-only.
    """
    if not batches:
        raise ValueError("batches must be non-empty")
    model.eval()
    losses: list[torch.Tensor] = []
    for batch in batches:
        ids = batch["input_ids"].to(device)
        attn = batch.get("attention_mask")
        attn_t = attn.to(device) if torch.is_tensor(attn) else None
        out = model(input_ids=ids, attention_mask=attn_t, labels=ids)
        if hasattr(out, "loss") and out.loss is not None:
            losses.append(out.loss.detach().float())
            continue
        logits = out.logits if hasattr(out, "logits") else model(input_ids=ids, attention_mask=attn_t).logits
        losses.append(mean_nll_from_logits(ids, logits).detach())

    stacked = torch.stack(losses).mean()
    return float(torch.exp(stacked).item())


@torch.no_grad()
def latency_ms_batches_mean(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    *,
    warmup: int = 3,
    reps_each: int = 5,
) -> float:
    """Average forward latency across windows."""
    if not batches:
        raise ValueError("batches must be non-empty")
    samples: list[float] = []
    for i, batch in enumerate(batches):
        wm = warmup if i == 0 else max(1, warmup // 3)
        samples.append(latency_ms_per_step(model, batch["input_ids"], warmup=wm, reps=reps_each))
    return float(sum(samples) / len(samples))
