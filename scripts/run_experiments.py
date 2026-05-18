#!/usr/bin/env python3
"""Full experiment matrix for the 260C report.

Produces, through ONE shared eval harness (same tokens, same metric):
  * sanity gate         – per-channel vs per-tensor INT8 quant-only
  * ablations           – dense / prune-only / quant-only / prune+quant / full
  * sweep               – bits x sparsity x fp16_fraction
  * routing ablation    – salience vs kind vs uniform vs random
  * literature baselines – Wanda-only, RTN-per-channel (AWQ/GPTQ-class)
  * memory / FLOPs       – theoretical bits-per-weight + FLOP reduction
Everything is dumped to results/experiments.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from saec.benchmarks import latency_ms_batches_mean, perplexity_hf_batches_mean  # noqa: E402
from saec.hf_layers import is_weight_matrix_module  # noqa: E402
from saec.pipeline import (  # noqa: E402
    CompressionPipelineConfig,
    compress_inplace,
    run_calibration,
    select_fp16_modules,
)
from saec.structure import classify_linear_name  # noqa: E402


def model_max_positions(config, fallback=1024):
    for attr in ("n_positions", "max_position_embeddings"):
        v = getattr(config, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return fallback


def build_windows(tokenizer, split, batch_size, seq_len, max_batches, config):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [t for t in ds["text"] if t and len(t.strip()) > 80]
    ctx = min(seq_len, model_max_positions(config))
    cap = min(len(texts), max(batch_size * max_batches * 64 + 1536, 3072))
    ids: list[int] = []
    for para in texts[:cap]:
        ids.extend(
            tokenizer(para, add_special_tokens=False, truncation=True, max_length=ctx)["input_ids"]
        )
    flat = torch.tensor(ids, dtype=torch.long)
    usable = flat.numel() // seq_len * seq_len
    flat = flat[:usable]
    out = []
    idx = 0
    while idx + batch_size * seq_len <= usable and len(out) < max_batches:
        b = flat[idx : idx + batch_size * seq_len].reshape(batch_size, seq_len)
        idx += batch_size * seq_len
        out.append({"input_ids": b, "attention_mask": torch.ones_like(b)})
    return out


def dense_param_table(model):
    """Per-dense-layer param count + classified kind."""
    tbl = {}
    for n, m in model.named_modules():
        if is_weight_matrix_module(m):
            tbl[n] = {"params": int(m.weight.numel()), "kind": classify_linear_name(n).value}
    return tbl


def theoretical_memory(tbl, fp16_keep, bits, sparsity, group_size=0):
    """Mean bits/weight and compression vs an FP16 model.

    Quantized layers stored at ``bits``; FP16 keep-set at 16. Unstructured
    sparsity is credited optimistically (nonzeros only) AND with a realistic
    CSR-style 1-bit/elt index overhead, so the report can cite both.
    """
    tot = sum(v["params"] for v in tbl.values())
    # Group-wise quant stores one fp16 scale per group: +16/group_size bits/wt.
    scale_oh = (16.0 / group_size) if group_size and group_size > 0 else 0.0
    bits_sum = 0.0
    q_params = 0
    for n, v in tbl.items():
        if n in fp16_keep:
            b = 16.0
        else:
            b = bits + scale_oh
            q_params += v["params"]
        bits_sum += v["params"] * b
    mean_bits = bits_sum / tot
    keep = 1.0 - sparsity
    return {
        "dense_params": tot,
        "quantized_params": q_params,
        "quantized_share_pct": round(100.0 * q_params / tot, 2),
        "mean_bits_per_weight_no_sparsity": round(mean_bits, 4),
        "compression_vs_fp16_no_sparsity": round(16.0 / mean_bits, 3),
        "mean_bits_per_weight_ideal_sparse": round(mean_bits * keep, 4),
        "mean_bits_per_weight_csr_sparse": round(mean_bits * keep + 1.0 * keep, 4),
        "compression_vs_fp16_csr_sparse": round(16.0 / (mean_bits * keep + 1.0 * keep), 3),
    }


def theoretical_flops(tbl, sparsity, seq_len):
    """Per-window dense GEMM MACs and the sparsity-credited reduction.

    Unstructured pruning yields NO speedup on dense kernels (reported as a
    disclaimer); this is the *theoretical* ceiling a sparse kernel could hit.
    """
    macs = sum(v["params"] for v in tbl.values()) * seq_len
    return {
        "gemm_macs_per_window": int(macs),
        "theoretical_macs_pruned": int(macs * (1.0 - sparsity)),
        "theoretical_flop_reduction_pct": round(100.0 * sparsity, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="gpt2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--calib-batches", type=int, default=64)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="gpt2")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)
    model.eval()
    orig_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    seq = max(16, min(args.seq_len, model_max_positions(model.config)))

    calib = build_windows(tok, "train", args.batch_size, seq, args.calib_batches, model.config)
    eval_w = build_windows(tok, "validation", args.batch_size, seq, args.eval_batches, model.config)
    n_windows = len(eval_w)
    act_norms = run_calibration(model, iter(calib), device=device, max_batches=len(calib))

    tbl = dense_param_table(model)
    n_dense = len(tbl)

    def reset():
        model.load_state_dict({k: v.to(device) for k, v in orig_state.items()})
        model.eval()

    def evaluate():
        with torch.no_grad():
            ppl = perplexity_hf_batches_mean(model, eval_w, device=device)
        lat = latency_ms_batches_mean(model, eval_w, warmup=args.warmup, reps_each=args.reps)
        return ppl, lat

    # ---- baseline ----
    reset()
    base_ppl, base_lat = evaluate()
    print(f"[baseline] ppl={base_ppl:.3f} lat={base_lat:.2f}ms")

    results = {
        "meta": {
            "model": args.model_name,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "dense_layers": n_dense,
            "eval_windows": n_windows,
            "tokens_per_window": args.batch_size * seq,
            "total_eval_tokens": n_windows * args.batch_size * seq,
            "seq_len": seq,
            "calib_batches": len(calib),
            "seed": args.seed,
        },
        "baseline": {"ppl": base_ppl, "lat_ms": base_lat},
        "runs": [],
    }

    def run(name, *, sparsity, bits, fp16_frac, routing="salience", group_size=128,
            do_prune=True, do_quant=True, per_channel=True, per_output=True):
        reset()
        cfg = CompressionPipelineConfig(
            prune_sparsity=sparsity, low_bitwidth=bits, fp16_fraction=fp16_frac,
            routing=routing, routing_seed=args.seed,
            per_channel_quant=per_channel, per_output_prune=per_output,
            group_size=group_size,
        )
        t0 = time.time()
        fp16_keep, _ = compress_inplace(
            model, act_norms, cfg, device=device, do_prune=do_prune, do_quant=do_quant
        )
        ppl, lat = evaluate()
        mem = theoretical_memory(
            tbl, fp16_keep if do_quant else set(tbl),
            bits if do_quant else 16, sparsity if do_prune else 0.0,
            group_size if do_quant and per_channel else 0,
        )
        rec = {
            "name": name, "sparsity": sparsity if do_prune else 0.0,
            "bits": bits if do_quant else 16, "fp16_frac": fp16_frac if do_quant else 1.0,
            "routing": routing, "do_prune": do_prune, "do_quant": do_quant,
            "per_channel": per_channel, "per_output": per_output,
            "group_size": group_size if do_quant and per_channel else 0,
            "fp16_keep_n": len(fp16_keep),
            "ppl": round(ppl, 4), "ppl_delta_pct": round(100 * (ppl - base_ppl) / base_ppl, 3),
            "lat_ms": round(lat, 3), "lat_delta_pct": round(100 * (lat - base_lat) / base_lat, 3),
            "mem": mem,
            "flops": theoretical_flops(tbl, sparsity if do_prune else 0.0, seq),
            "secs": round(time.time() - t0, 1),
        }
        results["runs"].append(rec)
        print(f"[{name}] ppl={ppl:.3f} ({rec['ppl_delta_pct']:+.1f}%) "
              f"bits/w={mem['mean_bits_per_weight_no_sparsity']} lat={lat:.1f}ms")
        return rec

    # ---- 1. sanity gate: the actual bug fix (INT8, no groups) ----
    run("sanity:int8_per_tensor(legacy-bug)", sparsity=0, bits=8, fp16_frac=0.0,
        do_prune=False, per_channel=False, group_size=0)
    run("sanity:int8_per_channel(fixed)", sparsity=0, bits=8, fp16_frac=0.0,
        do_prune=False, group_size=0)

    # ---- 2. quantizer-granularity study (quant-only, the core fix) ----
    for b in (4, 3):
        run(f"grain:int{b}_per_tensor", sparsity=0, bits=b, fp16_frac=0.0,
            do_prune=False, per_channel=False, group_size=0)
        run(f"grain:int{b}_per_channel", sparsity=0, bits=b, fp16_frac=0.0,
            do_prune=False, group_size=0)
        run(f"grain:int{b}_group128", sparsity=0, bits=b, fp16_frac=0.0,
            do_prune=False, group_size=128)
        run(f"grain:int{b}_group64", sparsity=0, bits=b, fp16_frac=0.0,
            do_prune=False, group_size=64)

    # ---- 3. ablations @ INT4 g128 / 50% / 5% salient ----
    AB_BITS, AB_SP, AB_FP = 4, 0.5, 0.05
    run("ablate:prune_only@0.5", sparsity=AB_SP, bits=AB_BITS, fp16_frac=0.0, do_quant=False)
    run("ablate:quant_only@int4g128", sparsity=0, bits=AB_BITS, fp16_frac=0.0, do_prune=False)
    run("ablate:prune+quant(no_fp16)", sparsity=AB_SP, bits=AB_BITS, fp16_frac=0.0)
    run("ablate:full(prune+quant+5%fp16-salience)", sparsity=AB_SP, bits=AB_BITS, fp16_frac=AB_FP)

    # ---- 4. sweep (group-128) ----
    for b in (8, 4, 3):
        run(f"sweep:bits{b}@sp0.5@fp5", sparsity=0.5, bits=b, fp16_frac=0.05)
    for sp in (0.0, 0.3, 0.5):
        run(f"sweep:sp{sp}@int4@fp5", sparsity=sp, bits=4, fp16_frac=0.05)
    for fp in (0.0, 0.02, 0.05, 0.10):
        run(f"sweep:fp{fp}@int3@sp0.5", sparsity=0.5, bits=3, fp16_frac=fp)

    # ---- 5. routing ablation @ INT3 g128 / 50% / 5% ----
    for r in ("salience", "kind", "uniform", "random"):
        run(f"routing:{r}@int3", sparsity=0.5, bits=3, fp16_frac=0.05, routing=r)

    # ---- 6. literature baselines (same harness) ----
    run("baseline:wanda_only@0.5", sparsity=0.5, bits=4, fp16_frac=0.0, do_quant=False)
    run("baseline:rtn_per_channel_int4", sparsity=0, bits=4, fp16_frac=0.0,
        do_prune=False, group_size=0)
    run("baseline:gptq_class_int4_g128", sparsity=0, bits=4, fp16_frac=0.0,
        do_prune=False, group_size=128)
    run("baseline:gptq_class_int3_g128", sparsity=0, bits=3, fp16_frac=0.0,
        do_prune=False, group_size=128)

    # ---- per-layer routing map (salience @ INT3, 10% keep) ----
    reset()
    cfg = CompressionPipelineConfig(prune_sparsity=0.5, low_bitwidth=3, fp16_fraction=0.10,
                                    routing="salience", routing_seed=args.seed, group_size=128)
    keep_sal = select_fp16_modules(model, act_norms, cfg)
    layer_map = []
    for n, v in tbl.items():
        layer_map.append({
            "layer": n, "kind": v["kind"], "params": v["params"],
            "protected_salience": n in keep_sal,
        })
    results["layer_map"] = layer_map
    results["meta"]["gpu_mem_note"] = (
        "QDQ keeps weights as float tensors, so torch.cuda memory is unchanged; "
        "the bits/weight column is the deployable figure."
    )

    out = _REPO / "results"
    out.mkdir(exist_ok=True)
    fp = out / f"experiments_{args.tag}.json"
    fp.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
