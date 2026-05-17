# Structure-Aware Ensembling of Pruning and Quantization — Stage 1: Activation-Aware Pruning

Course research code for the 260C project *Structure-Aware Ensembling of Pruning and
Quantization*. The system is built in two stages:

- **Stage 1 (this commit): activation-aware pruning.** Calibration of incoming-activation
  RMS statistics, Wanda-style per-output-neuron unstructured pruning, with the token
  embedding and weight-tied output head kept dense. Includes the shared perplexity /
  latency evaluation harness.
- **Stage 2 (forthcoming): mixed-precision quantization + structure-aware routing.**
  Per-channel / group-wise low-bit weight QDQ and the salient-layer FP16 keep-set,
  ensembled on top of Stage 1.

## Stage 1 contents

| Component | Location |
|-----------|----------|
| Calibration (incoming-activation RMS) | `src/saec/calibration.py` |
| Wanda pruning (`|W| ⊙ act-RMS`, per-output group) | `src/saec/wanda.py` |
| Layer-type tagging / embedding-head detection | `src/saec/structure.py` |
| HF layer detection (Linear + GPT-2 Conv1D) | `src/saec/hf_layers.py` |
| Pruning pipeline (calibrate → prune, head kept dense) | `src/saec/pipeline.py` |
| Evaluation (perplexity + latency) | `src/saec/benchmarks.py`, `scripts/run_pipeline.py` |

## Setup

```bash
cd 260c-structure-aware-compression
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_pipeline.py --model-name gpt2 --device cpu --calib-batches 8 --sparsity 0.5
# With GPU:
python scripts/run_pipeline.py --model-name gpt2 --device cuda \
  --calib-batches 24 --batch-size 4 --seq-len 256 --sparsity 0.5
```

WikiText-2 is downloaded automatically via `datasets`.

## Notes / limitations

- The embedding and (weight-tied) output head are never pruned — pruning the tied matrix
  corrupts the input embedding. This follows standard practice in Wanda / GPTQ / AWQ.
- Unstructured pruning does not accelerate standard dense GEMM; real speedups require
  sparse kernels or structured/N:M patterns. End-to-end timing is reported on standard
  PyTorch ops for methodology reproduction only.
- Scaling to **Qwen2.5**-class stacks: the hooks target `nn.Linear` everywhere; extend
  `hf_layers.weight_and_in_dim` if a custom GEMM wrapper appears and widen
  `classify_linear_name` for new module prefixes.
