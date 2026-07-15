# FlexAttention in ONNX

This repository uses a separately maintained, pinned ONNX Runtime fork and a
canonical PrefixLM model contract for direct execution of
`ai.onnx.preview::FlexAttention` version 1. Clone the fork URL and check out the
commit recorded in [`runtime-pins.json`](runtime-pins.json) as
`onnxruntime-src/`. The direct CPU and CUDA kernels keep the original
FlexAttention node in the optimized graph and do not allocate a
`[B, H, L, S]` score tensor.

The supported modifier is:

```text
q_abs = q_start[b] + q
k_abs = kv_start[b] + k
allowed = (k_abs < prefix_len[b]) OR (k_abs <= q_abs)
```

`prefix_len`, `q_start`, and `kv_start` are `int64[B]` outer-scope
captures. The public model has only Q, K, and V as explicit FlexAttention
inputs. Non-identity `prob_mod`, padding, segments, ALiBi, sliding windows, and
soft caps are intentionally rejected.

## Export

Use Python 3.11 and install the direct-script dependencies with `uv`:

```bash
uv venv --python 3.11
uv pip install numpy onnx torch ml-dtypes
```

```bash
uv run python prefixlm/export_canonical_flexattention.py \
  --output /tmp/prefixlm.onnx \
  --dtype float32
```

## Build the CPU runtime

```bash
uv run python onnxruntime-src/tools/ci_build/build.py \
  --config RelWithDebInfo \
  --build_dir build-ort-prefixlm \
  --update --build --parallel 4 \
  --build_shared_lib --build_wheel --skip_tests \
  --enable_preview_flex_attention
```

The preview option is OFF by default. It registers the preview schema and the
native CPU kernel only when explicitly enabled.

To verify numerical behavior, node retention, assignment, and profiling with the
built wheel:

```bash
uv pip install --target /tmp/ort-prefixlm --no-deps \
  build-ort-prefixlm/RelWithDebInfo/dist/onnxruntime-*.whl
PYTHONPATH=/tmp/ort-prefixlm \
  uv run python prefixlm/verify_ort_runtime.py --dtype float32
PYTHONPATH=/tmp/ort-prefixlm \
  uv run python prefixlm/verify_ort_runtime.py --dtype float16
PYTHONPATH=/tmp/ort-prefixlm \
  uv run python prefixlm/verify_ort_runtime.py \
    --dtype bfloat16 --skip-expanded
PYTHONPATH=/tmp/ort-prefixlm \
  uv run python prefixlm/verify_modifier_parsers.py
```

## Build the CUDA Plugin EP

CUDA requires SM80 or newer. The direct plugin kernel supports fp16 and bf16,
head sizes divisible by 8, and head sizes up to 1024. It extends ORT's CUTLASS
memory-efficient attention with a PrefixLM mask, BNSH Q/K/V/output strides,
GQA head mapping, and device-side position arrays. It does not materialize
scores or replicate K/V heads. For `Dv > 128`, CUTLASS uses a float32 output
accumulator of `B*Hq*L*Dv` elements; smaller V heads need no output workspace.

```bash
uv run python onnxruntime-src/tools/ci_build/build.py \
  --config RelWithDebInfo \
  --build_dir build-ort-prefixlm-cuda \
  --update --build --parallel 4 --skip_tests \
  --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr \
  --enable_preview_flex_attention \
  --cmake_extra_defines \
    onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON \
    CMAKE_CUDA_ARCHITECTURES=80
```

The wheel and CUDA Plugin EP library must come from the same source revision.
Set `CMAKE_CUDA_ARCHITECTURES` to the deployed GPU architecture; the example
uses SM80.
Run the CUDA verifier by passing the resulting plugin to `--ep-library`, as
shown in [`prefixlm/README.md`](prefixlm/README.md).
Exact ORT, ONNX schema, and plugin API revisions are recorded in
[`runtime-pins.json`](runtime-pins.json). See
[`prefixlm/README.md`](prefixlm/README.md) for the file map and test commands.
