# FlexAttention in ONNX

Examples for exporting PyTorch FlexAttention to ONNX as
`ai.onnx.preview::FlexAttention` and running it with a preview ONNX Runtime CPU
EP build.

## Requirements

- `uv`
- ccache

## Environment

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch onnxscript flatbuffers
```

## Build and install the latest ONNX package

```bash
git clone --recursive https://github.com/onnx/onnx onnx-src
cd onnx-src
uv pip install -v .
cd ..
```

## Build and install ONNX Runtime with FlexAttention support

```bash
git clone --recursive -b preview-flexattention-cpu https://github.com/mshr-h/onnxruntime onnxruntime-src
cd onnxruntime-src

./build.sh \
  --config Release \
  --update \
  --build \
  --use_cache \
  --parallel \
  --build_wheel \
  --skip_tests \
  --compile_no_warning_as_error

uv pip install --force-reinstall --no-deps build/Linux/Release/dist/onnxruntime-*.whl

cd ..
```

## Examples

See [`prefixlm/README.md`](prefixlm/README.md) for the PrefixLM-style
FlexAttention export and runtime commands.
