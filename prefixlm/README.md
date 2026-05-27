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
