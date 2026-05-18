# Results

All numbers are produced by `scripts/run_experiments.py` and stored in
`results/experiments_gpt2.json` and `results/experiments_qwen2.5-0.5b.json`.

## Experimental setup

We evaluate on two decoder-only language models: GPT-2 (124 M parameters, 49 dense
Linear/Conv1D layers) as the primary model, and Qwen2.5-0.5B (0.5 B parameters, 169 dense
Linear layers) as a transfer target from the proposed Qwen2.5 family (Qwen2.5-Omni itself
exceeds single-GPU memory and cannot be calibrated on the Tesla T4 used here). Both models run
in fp32 on a Tesla T4.

Calibration uses WikiText-2 train (64 windows for GPT-2, 32 for Qwen2.5-0.5B). Evaluation uses
WikiText-2 validation, with perplexity defined as `exp(mean per-token NLL)` over 40 windows
(40 960 tokens) for GPT-2 and 20 windows (20 480 tokens) for Qwen2.5-0.5B, at sequence length
256, seed 42. Every compressed configuration is evaluated on the identical token windows and
metric as its baseline, so all reported deltas are like-for-like. Baseline perplexity is
**40.40** for GPT-2 and **15.00** for Qwen2.5-0.5B. Memory is reported as effective
bits-per-weight (16 for FP16-kept layers, `bits + 16/group` for quantized layers); unstructured
sparsity is additionally credited with a CSR-style ≈1-bit/element index overhead. The embedding
and the weight-tied output head are always kept in full precision, following standard practice
in Wanda, GPTQ and AWQ.

## 1. Effect of quantization granularity

We first isolate the effect of the quantization grouping granularity, with no pruning and no
FP16 keep-set, sweeping per-tensor, per-output-channel, group-128 and group-64 scales.

**Δ perplexity vs baseline:**

| Bits | per-tensor | per-channel | group-128 | group-64 |
|---|---|---|---|---|
| INT8 — GPT-2 | +14.7 % | **−0.2 %** | — | — |
| INT8 — Qwen2.5-0.5B | +7.1 % | **+0.3 %** | — | — |
| INT4 — GPT-2 | +19 598 % | +49.1 % | +17.1 % | **+16.8 %** |
| INT4 — Qwen2.5-0.5B | +5.9×10⁸ % | +116 % | +39.3 % | **+26.5 %** |
| INT3 — GPT-2 | +11 544 % | +4 120 % | +446 % | **+241 %** |
| INT3 — Qwen2.5-0.5B | +3.4×10⁶ % | +1.2×10⁶ % | +1 108 % | **+562 %** |

Granularity is the dominant factor for low-bit weight quantization. At INT8, per-channel
quantization is effectively lossless on both models (±0.3 %), whereas per-tensor scaling already
costs +7–15 %. Below 8 bits, per-tensor RTN collapses entirely. With group-wise quantization
(the recipe used by AWQ/GPTQ/llama.cpp), INT4 becomes viable — group-64 costs only +16.8 % on
GPT-2 and +26.5 % on Qwen2.5-0.5B at ≈7.5–7.9 effective bits/weight. INT3 remains expensive even
at group-64 (+241 % / +562 %): plain round-to-nearest provides no error compensation, which is
the gap that motivates error-aware quantizers and the salient-weight protection studied below.

## 2. Component ablations

We ablate pruning and quantization individually and jointly at a fixed operating point
(INT4, group-128, 50 % Wanda sparsity, 5 % salient-FP16 keep-set), on the same evaluation.

| Configuration | GPT-2 Δppl | Qwen Δppl | dense bits/wt | CSR-sparse bits/wt | comp. vs FP16 |
|---|---|---|---|---|---|
| Prune-only (Wanda 50 %) | +54.2 % | +58.3 % | 16.0 | 8.50 | 1.9× |
| Quant-only (INT4 g128) | +17.1 % | +39.3 % | 7.84 | 8.84 | 1.8× |
| Prune + Quant (no FP16) | +91.3 % | +134.5 % | 7.84 | 4.42 | 3.6× |
| Full (+5 % salient FP16) | +90.1 % | +129.9 % | 8.29 | 4.64 | 3.4× |

The two techniques are non-additive and interact adversely: 50 % Wanda alone (+54 %) composed
with INT4 alone (+17 %) yields +91 %, worse than the sum — quantizing the survivors of pruning
compounds error. At these model scales, 50 % unstructured pruning is the dominant error term,
not quantization; Wanda's near-lossless 50 % behavior is a model-scale phenomenon that does not
hold on 0.1–0.5 B models. The 5 % salient-FP16 keep-set changes perplexity only marginally
(+91.3 → +90.1 % on GPT-2).

## 3. Sweeps

**Bit-width @ 50 % sparsity, 5 % FP16 (Δppl):**

| | INT8 | INT4 | INT3 |
|---|---|---|---|
| GPT-2 | +54.1 % | +90.1 % | +704.9 % |
| Qwen2.5-0.5B | +58.8 % | +129.9 % | +1 747.9 % |

**Sparsity @ INT4, 5 % FP16 (Δppl):**

| | 0 % | 30 % | 50 % |
|---|---|---|---|
| GPT-2 | +16.3 % | +21.4 % | +90.1 % |
| Qwen2.5-0.5B | +35.5 % | +43.3 % | +129.9 % |

**FP16-keep fraction @ INT3, 50 % sparsity (Δppl):**

| keep | 0 % | 2 % | 5 % | 10 % |
|---|---|---|---|---|
| GPT-2 | +721.7 % | +707.8 % | +704.9 % | +625.6 % |
| Qwen2.5-0.5B | +1 894 % | +1 808 % | +1 748 % | +1 558 % |

The bit-width sweep shows INT8 safe, INT4 moderate, and a sharp INT3 cliff on both models. The
sparsity sweep reveals that 30 % unstructured sparsity is nearly free on top of INT4 while 50 %
is a cliff, giving a concrete operating-point guideline of ≤30 % sparsity for sub-1 B models.
Increasing the FP16-keep fraction yields only a small, monotone improvement and does not rescue
INT3 — a 2–10 % escape hatch is insufficient at 3 bits.

## 4. Routing ablation

We compare four policies for selecting the FP16 keep-set, holding bitwidth (INT3), sparsity
(50 %) and budget (5 %) fixed: **salience** (largest activation-weighted quant error
Σ act_rms²·(ΔW)²), **kind** (attention/head-first structural prior), **uniform** (fixed
name-sorted slice), and **random** (seeded).

| Routing | GPT-2 Δppl | Qwen2.5-0.5B Δppl |
|---|---|---|
| salience | +704.9 % | +1 747.9 % |
| kind | +621.6 % | +806.9 % |
| uniform | +621.6 % | +798.8 % |
| random | +669.1 % | +1 647.6 % |

The activation-salience criterion does not outperform the alternatives; the structural prior and
even the structure-blind uniform slice are better on both models, substantially so on
Qwen2.5-0.5B (+807 % vs +1 748 %). At a small FP16 budget in a severe INT3 regime, the choice of
which few layers to protect is second-order relative to bitwidth and sparsity, and the greedy
per-layer salience estimator — which ignores inter-layer interaction — mis-ranks layers. The
effective structural decision is the always-protected embedding/head; the learned router does
not add value in this regime.

## 5. Per-layer structure analysis

Under the salience criterion with a 10 % keep budget, the protected layers are:

- **GPT-2** (6 of 49): `h.3.attn.c_attn`, `h.4.attn.c_attn`, `h.10.mlp.c_proj`,
  `h.11.attn.c_proj`, `h.11.mlp.c_proj`, `lm_head` — a mix of early attention and late-block
  MLP/attention output projections, not a uniform "all attention" pattern.
- **Qwen2.5-0.5B** (18 of 169): `lm_head` plus MLP `gate_proj`/`up_proj` concentrated in
  layers 16–23 (the final third of the network).

The routing is layer-discriminative and produces an interpretable pattern — late-block MLP
projections are consistently flagged as most quantization-sensitive — even though, as Section 4
shows, this pattern is not more effective than a structural prior at INT3 with a 5 % budget.

## 6. Memory footprint

| Configuration (GPT-2) | dense bits/wt | CSR-sparse bits/wt | comp. vs FP16 |
|---|---|---|---|
| INT4 g128, dense | 7.84 | — | 2.04× |
| INT4 g128 + 50 % Wanda | 7.84 | 4.42 | 3.62× |
| INT3 g128 + 50 % Wanda | 7.15 | 4.07 | 3.93× |
| Full (INT4 g128, 50 %, 5 % FP16) | 8.29 | 4.64 | 3.44× |

The combined pipeline reaches 3.4–3.9× theoretical compression versus an FP16 model. GPT-2's
weight-tied 38 M-parameter head (≈31 % of weights) is kept in full precision, which floors the
dense bits/weight near 7–8 even at INT3. Measured GPU memory is unchanged because the simulated
QDQ stores dequantized float weights; the bits/weight column is the deployable figure and would
require packed INT kernels to realize.

## 7. Latency

| | baseline | compressed (typical) | Δ |
|---|---|---|---|
| GPT-2 | 81.6 ms | 98–103 ms | +20 – 24 % |
| Qwen2.5-0.5B | 431 ms | 445–451 ms | +3 – 5 % |

All compressed configurations are slower than the baseline. Unstructured pruning still issues
dense GEMMs (zeros are multiplied like any other weight), and simulated QDQ stores dequantized
float weights, so there is no kernel-level acceleration — only added quantize/dequantize work.
We therefore report the achievable speedup as a theoretical quantity: the FLOP reduction equals
the sparsity (0 % at dense, 30 % at sp0.3, 50 % at sp0.5), realizable only with sparse or
low-bit kernels (sparse CUTLASS, bitsandbytes, AutoGPTQ), which are out of scope here. No
wall-clock speedup is claimed.

## 8. Summary

Granularity is the dominant factor in low-bit quantization quality: INT8 per-channel is free,
INT4 is usable with group-wise quantization (+17 % GPT-2 / +27 % Qwen2.5-0.5B), and INT3 remains
expensive (+241 % / +562 % at best) with a small FP16 budget unable to recover it. Pruning is
the costlier half of the ensemble at these scales — 50 % Wanda alone costs +54–58 % and 50 % is
a sharp cliff, while 30 % is nearly free; the literature claim of near-lossless 50 % pruning does
not transfer to sub-1 B models. The two techniques interact adversely rather than additively.
The activation-salience router is interpretable but does not outperform a trivial structural
prior in the hardest regime; the decisive structure-aware element is the always-protected
embedding/output head. Latency is reframed as a theoretical FLOP/bit reduction. All conclusions
hold consistently on both GPT-2 and the proposed-family Qwen2.5-0.5B.
