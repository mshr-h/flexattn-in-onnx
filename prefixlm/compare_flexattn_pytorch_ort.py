from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import torch

from export_flexattn_onnx import (
    MAIN_OPSET,
    PREVIEW_DOMAIN,
    annotate_score_mod_types,
    export_model_proto,
    verify_model,
)
from model import build_model, torch_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a FlexAttention ONNX model in memory and compare ORT CPU EP with PyTorch.",
    )
    parser.add_argument("--batch", type=int, default=66)
    parser.add_argument("--seq-len", type=int, default=379)
    parser.add_argument("--prefix-len", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--input-seed", type=int, default=0)
    parser.add_argument(
        "--opset",
        type=int,
        default=MAIN_OPSET,
        help="Main ONNX opset for torch.onnx.export(dynamo=True).",
    )
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if not 0 <= args.prefix_len <= args.seq_len:
        raise ValueError("--prefix-len must satisfy 0 <= prefix_len <= seq_len")
    if args.embed_dim <= 0:
        raise ValueError("--embed-dim must be positive")
    if args.num_heads <= 0:
        raise ValueError("--num-heads must be positive")
    if args.embed_dim % args.num_heads != 0:
        raise ValueError("--embed-dim must be divisible by --num-heads")
    if args.opset < 18:
        raise ValueError("--opset must be at least 18 for the dynamo exporter path")


def numpy_dtype(dtype: str) -> np.dtype:
    if dtype == "float16":
        return np.dtype(np.float16)
    if dtype == "float32":
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported dtype: {dtype}")


def default_tolerances(dtype: str) -> tuple[float, float]:
    if dtype == "float16":
        return 5e-2, 5e-2
    return 1e-4, 1e-4


def count_flex_attention_nodes(model_proto) -> int:
    return sum(
        1
        for node in model_proto.graph.node
        if node.domain == PREVIEW_DOMAIN and node.op_type == "FlexAttention"
    )


def compare_outputs(
    torch_output: np.ndarray, ort_output: np.ndarray, rtol: float, atol: float
) -> bool:
    torch_f32 = torch_output.astype(np.float32)
    ort_f32 = ort_output.astype(np.float32)
    diff = np.abs(torch_f32 - ort_f32)
    rel = diff / np.maximum(np.abs(torch_f32), np.float32(1e-12))

    torch_finite = bool(np.isfinite(torch_f32).all())
    ort_finite = bool(np.isfinite(ort_f32).all())
    allclose = bool(np.allclose(ort_f32, torch_f32, rtol=rtol, atol=atol))

    print(
        f"PyTorch output: shape={torch_output.shape} dtype={torch_output.dtype} finite={torch_finite}"
    )
    print(
        f"ORT output:     shape={ort_output.shape} dtype={ort_output.dtype} finite={ort_finite}"
    )
    print(f"max_abs_diff={float(diff.max())}")
    print(f"mean_abs_diff={float(diff.mean())}")
    print(f"max_rel_diff={float(rel.max())}")
    print(f"allclose={allclose} rtol={rtol} atol={atol}")

    ok = torch_finite and ort_finite and allclose
    if not ok:
        flat_index = int(np.argmax(diff))
        index = np.unravel_index(flat_index, diff.shape)
        print(
            "max_diff_at="
            f"{index} torch={float(torch_f32[index])} ort={float(ort_f32[index])} "
            f"abs_diff={float(diff[index])} rel_diff={float(rel[index])}"
        )
    return ok


def main() -> None:
    args = parse_args()
    validate_args(args)

    model = build_model(
        batch=args.batch,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        prefix_len=args.prefix_len,
        dtype=args.dtype,
        seed=args.model_seed,
    )

    input_shape = (args.batch, args.seq_len, args.embed_dim)
    x_np = (
        np.random.default_rng(args.input_seed)
        .standard_normal(input_shape)
        .astype(numpy_dtype(args.dtype))
    )
    x_torch = torch.from_numpy(x_np.copy()).to(dtype=torch_dtype(args.dtype))

    model_proto = export_model_proto(model, x_torch, args.opset)
    annotate_score_mod_types(model_proto)
    verify_model(model_proto)

    flex_count = count_flex_attention_nodes(model_proto)
    print(f"FlexAttention nodes: {flex_count}")

    session = ort.InferenceSession(
        model_proto.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    input_info = session.get_inputs()[0]
    ort_output = session.run(None, {input_info.name: x_np})[0]

    with torch.no_grad():
        torch_output = model(x_torch).detach().cpu().numpy()

    print(f"ONNX Runtime: {ort.__file__}")
    print(f"providers: {session.get_providers()}")
    print(f"input: {input_info.name} shape={x_np.shape} dtype={x_np.dtype}")

    default_rtol, default_atol = default_tolerances(args.dtype)
    rtol = default_rtol if args.rtol is None else args.rtol
    atol = default_atol if args.atol is None else args.atol
    if not compare_outputs(torch_output, ort_output, rtol=rtol, atol=atol):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
