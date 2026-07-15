from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto

try:
    from .canonical_flexattention import ModelConfig, build_model
    from .preflight_flexattention import validate_model
except ImportError:
    from canonical_flexattention import ModelConfig, build_model
    from preflight_flexattention import validate_model


DTYPES = {
    "float32": TensorProto.FLOAT,
    "float16": TensorProto.FLOAT16,
    "bfloat16": TensorProto.BFLOAT16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a canonical dynamic PrefixLM FlexAttention ONNX model."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--node-name", default="prefixlm_flex_attention")
    return parser.parse_args()


def export_model(path: Path, config: ModelConfig) -> onnx.ModelProto:
    model = build_model(config)
    validate_model(model)
    onnx.save(model, path)
    return model


def main() -> None:
    args = parse_args()
    export_model(
        args.output,
        ModelConfig(
            q_dtype=DTYPES[args.dtype], scale=args.scale, node_name=args.node_name
        ),
    )
    print(f"saved canonical PrefixLM FlexAttention model to {args.output}")


if __name__ == "__main__":
    main()
