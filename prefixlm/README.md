# Canonical PrefixLM FlexAttention

The canonical exporter creates one top-level
`ai.onnx.preview::FlexAttention` version 1 node. Q, K, and V use BNSH layout.
`prefix_len`, `q_start`, and `kv_start` remain graph inputs or initializers
and are captured by `score_mod`.

## Files

- `canonical_flexattention.py`: canonical score modifier, model builder, and
  NumPy reference
- `export_canonical_flexattention.py`: command-line exporter
- `preflight_flexattention.py`: strict model-contract validator
- `expanded_attention.py`: primitive MatMul/Softmax correctness oracle
- `verify_ort_runtime.py`: direct-kernel numerical, optimized-graph,
  profiling, and invalid-input checks for fp32/fp16/bf16
- `modifier_corpus.py`: accepted/rejected modifier models shared by all parsers
- `verify_modifier_parsers.py`: Python and provider-adapter corpus verifier
- `verify_cutlass_causal_regression.py`: existing causal CUTLASS path regression
- `benchmark_dtype_matrix.py`: CPU/CUDA dtype latency and isolated peak-memory matrix
- `benchmark_runtime.py`: model, dtype, I/O binding, and session helpers shared by benchmarks
- `benchmark_peak_memory.py`: fresh-process CPU RSS/CUDA allocator peak-memory worker
- `cuda_allocation.py`: allocator callbacks and per-window allocation state used by the matrix
- `test_canonical_flexattention.py`: exporter, parser-contract, and oracle tests
- `test_benchmark_dtype_matrix.py`: dtype, bf16 oracle, and allocation bookkeeping tests

The older PyTorch preview-export scripts are not part of the direct-kernel
contract. New integration and CI should use only the canonical files above.

## Export

```bash
uv run python prefixlm/export_canonical_flexattention.py \
  --output /tmp/prefixlm.onnx \
  --dtype float32
```

The accepted `score_mod` is the canonical DAG built from `Shape`, `Gather`,
`Range`, `Reshape`, `Add`, `Less`, `LessOrEqual`, `Or`, and
`Where`. The masked scalar is float32 negative infinity.

## Unit tests

```bash
uv run python -m unittest prefixlm.test_canonical_flexattention -v
```

## Direct-runtime verification

Use a wheel built with `--enable_preview_flex_attention`:

```bash
uv pip install --target /tmp/ort-prefixlm --no-deps \
  build-ort-prefixlm/RelWithDebInfo/dist/onnxruntime-*.whl
PYTHONPATH=/tmp/ort-prefixlm \
  uv run python prefixlm/verify_ort_runtime.py \
    --dtype float32 \
    --provider CPUExecutionProvider \
    --output-dir /tmp/prefixlm-verification
```

The verifier compares the direct kernel and expanded ONNX graph with the NumPy
reference for `P=0`, `P=S`, per-batch prefix lengths, prefill, decode,
continued prefill, GQA, `Dv != Dqk`, and offset KV caches. It also
asserts that:

- the optimized model retains the same-named FlexAttention node;
- no MatMul or Softmax expansion appears;
- each run produces exactly one FlexAttention profiling event;
- every event is assigned to the requested provider.
- negative positions, absolute-position overflow, empty K/V, and fully masked
  first queries are rejected.

Run the same modifier corpus through Python preflight and the CPU adapter:

```bash
PYTHONPATH=/tmp/ort-prefixlm \
  uv run python prefixlm/verify_modifier_parsers.py
```

For CUDA, register the Plugin EP library built from the same revision:

```bash
PYTHONPATH=/tmp/ort-prefixlm \
  uv run python prefixlm/verify_ort_runtime.py \
    --dtype float16 \
    --provider CUDAExecutionProvider \
    --ep-library \
      build-ort-prefixlm-cuda/RelWithDebInfo/libonnxruntime_providers_cuda.so

PYTHONPATH=/tmp/ort-prefixlm \
  uv run python prefixlm/verify_modifier_parsers.py \
    --provider CUDAExecutionProvider \
    --ep-library \
      build-ort-prefixlm-cuda/RelWithDebInfo/libonnxruntime_providers_cuda.so
```

The CUDA kernel extends ORT's CUTLASS memory-efficient attention with PrefixLM,
BNSH strides, GQA head mapping, and the three device-side position arrays. Run
the existing causal-mode regression with FlashAttention disabled so dispatch is
forced through the same CUTLASS implementation:

```bash
PYTHONPATH=/tmp/ort-prefixlm \
  uv run python prefixlm/verify_cutlass_causal_regression.py \
    --ep-library \
      build-ort-prefixlm-cuda/RelWithDebInfo/libonnxruntime_providers_cuda.so
```

The runtime verifier feeds bf16 values through OrtValue and I/O binding. For
CPU bf16 it casts the expanded oracle inputs to float32 before MatMul and casts
the result back to bf16:

```bash
PYTHONPATH=/tmp/ort-prefixlm \
  uv run python prefixlm/verify_ort_runtime.py \
    --dtype bfloat16 \
    --provider CPUExecutionProvider
```

## Benchmark

Run the full dtype matrix for CPU fp32/fp16/bf16 and CUDA fp16/bf16. CUDA
latency is reported both with CPU-to-device transfers and with pre-bound device
tensors. CPU bf16 uses an explicitly labeled `expanded_fp32_cast` baseline,
because the CPU EP has no primitive bf16 `Expand` or `MatMul` kernel.

Run the CPU matrix without loading a CUDA Plugin EP:

```bash
PYTHONPATH=/tmp/ort-prefixlm \
  uv run python prefixlm/benchmark_dtype_matrix.py \
    --providers cpu \
    --seq-lens 128 512 1024 2048 4096 8192 \
    --warmup 20 \
    --repeats 100 \
    --cpu-workers 20 \
    --out-dir prefixlm/benchmark_results/cpu_block_gemm
```

Run CPU and CUDA together by selecting both providers and supplying the plugin:

```bash
PYTHONPATH=/tmp/ort-prefixlm \
  uv run python prefixlm/benchmark_dtype_matrix.py \
    --providers cpu cuda \
    --seq-lens 128 512 1024 2048 4096 8192 \
    --warmup 20 \
    --repeats 100 \
    --cpu-workers 20 \
    --ep-library \
      build-ort-prefixlm-cuda/RelWithDebInfo/libonnxruntime_providers_cuda.so
```

Each memory case runs in a fresh process. CPU memory is the Linux process
`VmHWM` total and its increase from the post-import baseline. CUDA allocation
history and its peak window are reset after warmup. The matrix reports the
measured peak, post-warmup incremental peak, largest measured allocation,
float32 score-buffer size, and score-buffer gate. CUDA context/library memory,
registers, and shared memory are not included in the external-allocator metric.
Results are written as JSON, latency CSV, memory CSV, and a Markdown summary.

## Failure behavior

The preflight and provider parsers reject unsupported modifiers. The native CPU
kernel performs the same parsing during session initialization, so bypassing
preflight does not silently fall back to Function expansion. Runtime validation
rejects invalid rank, dtype, batch/head/sequence relationships, GQA ratios,
capture shapes or types, non-finite scale, negative positions,
absolute-position overflow, zero K/V length, and a fully masked first query.
