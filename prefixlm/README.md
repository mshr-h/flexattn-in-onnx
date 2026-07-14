# PrefixLM-style Attention

This example exports a PrefixLM-style PyTorch FlexAttention module to ONNX as
`ai.onnx.preview::FlexAttention`.

## Files

- `model.py`: attention module with QKV projection, RoPE on Q/K, and PrefixLM mask
- `export_flexattn_onnx.py`: exports and validates the ONNX model
- `run_flexattn_cpu_ep.py`: runs a saved model with ONNX Runtime CPU EP
- `compare_flexattn_pytorch_ort.py`: compares PyTorch and ORT outputs

## Commands

Run from the repo root.

```bash
uv run python export_flexattn_onnx.py --output multihead_attention_flexattn.onnx
uv run python run_flexattn_cpu_ep.py --onnx-path multihead_attention_flexattn.onnx
uv run python compare_flexattn_pytorch_ort.py
```

## Validation

The exporter checks that ONNX validation passes, there is exactly one preview
FlexAttention node, `score_mod` is annotated, and input/output shapes match.


## Benchmark

Run from the repo root to compare `ai.onnx.preview::FlexAttention` with an
expanded lower-level ONNX graph on ONNX Runtime CPU EP.

```bash
uv run python prefixlm/benchmark_flexattn_vs_expansion.py \
  --seq-lens 128 512 1024 2048 \
  --batch 1 \
  --embed-dim 128 \
  --num-heads 4 \
  --prefix-len 3 \
  --dtype float32 \
  --warmup 10 \
  --repeats 50 \
  --out-dir prefixlm/benchmark_results
```

The benchmark requires an ONNX Runtime build with
`ai.onnx.preview::FlexAttention` support. Set `PYTHONPATH` to a rebuilt ONNX
Runtime package when measuring local kernel changes. It writes
`benchmark_results.csv`, `benchmark_results.json`, and `benchmark_summary.md`
under `--out-dir`, including latency statistics, correctness metrics, ORT
metadata, provider, graph optimization level, and thread settings. Use
`--save-models` only when you want to persist generated ONNX files; use
`--enable-profiling` or `--save-optimized-models` when you need diagnostics.
