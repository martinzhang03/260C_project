# Structure-aware ensembling: Wanda pruning + mixed-precision QDQ

Course-style research code for *Structure-Aware Ensembling of Pruning and Quantization* (260C proposal). It implements:

1. **Calibration** – RMS norms of incoming activations for dense layers (`nn.Linear` and GPT-2 `Conv1D`) used as Wanda-style signals.
2. **Wanda pruning** – unstructured sparsity via `|W| ⊙ activation-RMS` importance.
3. **Structure-aware mixed precision** – prioritizes attention / LM-head dense maps when skipping simulated low-bit weight quantization; remaining layers run symmetric fixed-point **QDQ** (`--bits` controls effective width).

## Setup

```bash
cd 260c-structure-aware-compression
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Defaults skew mild (`bits=8`, modest sparsity) so perplexity tends to stay usable without tuning.

```bash
python scripts/run_pipeline.py --model-name gpt2 --device cpu --calib-batches 8
# With GPU:
python scripts/run_pipeline.py --model-name gpt2 --device cuda --calib-batches 16 --batch-size 2 --seq-len 128 --sparsity 0.12 --bits 8
```

Closer to aggressive INT3-style runs from the proposal (expect sharper perplexity loss unless calibration is large):

```bash
python scripts/run_pipeline.py --bits 4 --sparsity 0.35 --fp16-fraction 0.35 --calib-batches 24
```

WikiText-2 is downloaded automatically via `datasets`.

## Mapping to the proposal

| Proposal item | Location |
|---------------|----------|
| Activation-aware pruning (Wanda) | `src/saec/wanda.py` |
| Mixed precision routing | [`src/saec/mixed_precision_quant.py`](src/saec/mixed_precision_quant.py), [`src/saec/pipeline.py`](src/saec/pipeline.py) |
| Structure-aware grouping | [`src/saec/structure.py`](src/saec/structure.py) (`tag_dense_weight_modules`) |
| Evaluation (perplexity + latency) | `src/saec/benchmarks.py`, `scripts/run_pipeline.py` |

Scaling to **Qwen2.5-Omni** or newer stacks: reuse the same hooks (`nn.Linear` everywhere) or extend `hf_layers.weight_and_in_dim` if a custom GEMM-wrapper appears; widen `classify_linear_name` for new module prefixes.

## Limitations

- Unstructured pruning does not accelerate standard dense GEMM; real speedups need sparse kernels or structured/N:M patterns.
- Weight-only QDQ is a practical stand-in for mixed-precision kernels in a class project environment.
