#!/usr/bin/env python3
"""
Research demo: GPT-2 (or compatible HF causal LM) with
  calibration → Wanda pruning → structure-aware simulated INT{N} weight QDQ.

This is intentionally CPU/GPU-safe and avoids custom sparse kernels.

Example:
  python scripts/run_pipeline.py --model-name gpt2 --device cuda \\
    --calib-batches 8 --batch-size 4 --seq-len 128 --sparsity 0.12 --bits 8
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from saec.benchmarks import latency_ms_batches_mean, perplexity_hf_batches_mean  # noqa: E402
from saec.hf_layers import is_weight_matrix_module  # noqa: E402
from saec.pipeline import CompressionPipelineConfig, compress_inplace, run_calibration  # noqa: E402


def _model_max_positions(config, fallback: int = 1024) -> int:
    for attr in ("n_positions", "max_position_embeddings"):
        v = getattr(config, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return fallback


def build_wikitext_batches(
    *,
    tokenizer,
    split: str,
    batch_size: int,
    seq_len: int,
    max_batches: int,
    config,
):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [t for t in ds["text"] if t and len(t.strip()) > 80]

    ctx = min(seq_len, _model_max_positions(config))
    # Larger paragraph budgets stabilize activation RMS estimates for pruning.
    para_budget = batch_size * max_batches * 64 + 1536
    paragraph_cap = min(len(texts), max(para_budget, 3072))
    text_buf = texts[:paragraph_cap]

    token_ids: list[int] = []
    for paragraph in text_buf:
        chunk = tokenizer(
            paragraph,
            add_special_tokens=False,
            truncation=True,
            max_length=ctx,
        )["input_ids"]
        token_ids.extend(chunk)

    ids_flat = torch.tensor(token_ids, dtype=torch.long)
    # Need enough tokens even if truncation shortens stream
    usable = ids_flat.numel() // seq_len * seq_len
    if usable <= 0:
        raise RuntimeError("Not enough tokenized WikiText rows for seq_len/context window.")
    ids_flat = ids_flat[:usable]

    yielded = 0
    idx = 0
    device_for_batch_build = torch.device("cpu")

    while idx + batch_size * seq_len <= usable and yielded < max_batches:
        batch_ids = (
            ids_flat[idx : idx + batch_size * seq_len].reshape(batch_size, seq_len).to(device_for_batch_build)
        )
        idx += batch_size * seq_len
        yielded += 1
        attn = torch.ones_like(batch_ids, dtype=torch.long, device=batch_ids.device)
        yield {"input_ids": batch_ids, "attention_mask": attn}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--calib-split", type=str, default="train", choices=("train", "validation", "test"))
    p.add_argument("--eval-split", type=str, default="validation", choices=("train", "validation", "test"))
    p.add_argument("--calib-batches", type=int, default=24)
    p.add_argument("--eval-batches", type=int, default=24, help="Windows averaged for perplexity + latency.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--sparsity", type=float, default=0.12)
    p.add_argument(
        "--bits",
        type=int,
        default=8,
        help="Symmetric weight QDQ simulation bitwidth on non-FP16 layers (use ~3–4 only for exploratory runs).",
    )
    p.add_argument(
        "--fp16-fraction",
        type=float,
        default=0.45,
        help="Fraction of dense weight layers (Linear/Conv1D) kept unquantized.",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=12, help="Timing repetitions averaged within each eval window.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    base = base.to(device)
    compressed = copy.deepcopy(base).to(device)

    eff_seq_len = max(16, min(args.seq_len, _model_max_positions(base.config)))

    calib_iter = build_wikitext_batches(
        tokenizer=tokenizer,
        split=args.calib_split,
        batch_size=args.batch_size,
        seq_len=eff_seq_len,
        max_batches=args.calib_batches,
        config=base.config,
    )
    act_norms = run_calibration(base, iter(calib_iter), device=device, max_batches=args.calib_batches)

    cfg = CompressionPipelineConfig(
        prune_sparsity=args.sparsity,
        low_bitwidth=args.bits,
        fp16_fraction=args.fp16_fraction,
        prefer_attention_fp16=True,
    )
    fp16_keep, _ = compress_inplace(compressed, act_norms, cfg, device=device)

    eval_batches = list(
        build_wikitext_batches(
            tokenizer=tokenizer,
            split=args.eval_split,
            batch_size=args.batch_size,
            seq_len=eff_seq_len,
            max_batches=args.eval_batches,
            config=base.config,
        )
    )
    if not eval_batches:
        raise RuntimeError("Eval batch iterator was empty.")

    base.eval()
    compressed.eval()
    with torch.no_grad():
        ppl_base = perplexity_hf_batches_mean(base, eval_batches, device=device)
        ppl_cmp = perplexity_hf_batches_mean(compressed, eval_batches, device=device)

    lat_b = latency_ms_batches_mean(
        base,
        eval_batches,
        warmup=args.warmup,
        reps_each=args.reps,
    )
    lat_c = latency_ms_batches_mean(
        compressed,
        eval_batches,
        warmup=args.warmup,
        reps_each=args.reps,
    )

    dense = [n for n, m in base.named_modules() if is_weight_matrix_module(m)]
    print("=== 260C structure-aware compression demo ===")
    print(f"model={args.model_name} device={device} seq_len={eff_seq_len} (effective)")
    print(
        f"calibration: split={args.calib_split} batches={args.calib_batches} batch_size={args.batch_size} "
        f"| eval: split={args.eval_split} windows={len(eval_batches)} batch_size={args.batch_size}"
    )
    print(f"dense layers (Linear/Conv1D): {len(dense)} | kept unquantized (structure-aware): {len(fp16_keep)}")
    print(f"prune sparsity={args.sparsity:.3f} low-bit QDQ bits={args.bits} fp16_fraction={args.fp16_fraction:.3f}")
    print(f"mean perplexity (exp mean NLL across eval windows): baseline={ppl_base:.3f} compressed={ppl_cmp:.3f}")
    print(f"mean forward latency across eval windows (ms): baseline={lat_b:.3f} compressed={lat_c:.3f}")
    print(f"(latency reps/window≈{args.reps}, warmup first window={args.warmup})")
    print()
    print("Note: unstructured pruning does not speed up dense matmuls without sparse kernels;")
    print("this reproduces the *methodology* and reports end-to-end timing on standard PyTorch ops.")


if __name__ == "__main__":
    main()
