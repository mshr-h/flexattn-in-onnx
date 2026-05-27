from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import onnx
import torch
from onnx import TensorProto, checker, helper
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.onnx._internal.exporter import _core

from model import build_model, torch_dtype


PREVIEW_DOMAIN = "ai.onnx.preview"
PREVIEW_OPSET = 1
MAIN_OPSET = 26


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a causal FlexAttention module to ONNX with ai.onnx.preview::FlexAttention.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("multihead_attention_flexattn.onnx")
    )
    parser.add_argument("--batch", type=int, default=66)
    parser.add_argument("--seq-len", type=int, default=379)
    parser.add_argument("--prefix-len", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--opset",
        type=int,
        default=MAIN_OPSET,
        help="Main ONNX opset for torch.onnx.export(dynamo=True).",
    )
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


def _tensor_value(
    name: str, dtype: ir.DataType, shape: Sequence[int | str]
) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def _node(
    nodes: list[ir.Node],
    op_type: str,
    inputs: Sequence[ir.Value | None],
    *,
    name: str,
    dtype: ir.DataType | None = None,
    shape: Sequence[int | str] | None = None,
    attributes: dict[str, Any] | None = None,
    domain: str = "",
    version: int | None = MAIN_OPSET,
) -> ir.Value:
    output = ir.Value(
        name=name,
        type=ir.TensorType(dtype) if dtype is not None else None,
        shape=ir.Shape(shape) if shape is not None else None,
    )
    nodes.append(
        ir.Node(
            domain,
            op_type,
            inputs=inputs,
            attributes=ir.convenience.convert_attributes(attributes or {}),
            outputs=[output],
            version=version,
        )
    )
    return output


def _const(
    nodes: list[ir.Node],
    name: str,
    value: Any,
    dtype: ir.DataType,
    shape: Sequence[int | str] | None = None,
) -> ir.Value:
    return _node(
        nodes,
        "Constant",
        [],
        name=name,
        dtype=dtype,
        shape=shape,
        attributes={"value": ir.tensor(value, dtype=dtype)},
    )


def _function_call(
    nodes: list[ir.Node],
    function: ir.Function,
    inputs: Sequence[ir.Value],
    *,
    name: str,
    dtype: ir.DataType,
    shape: Sequence[int | str],
) -> ir.Value:
    output = _tensor_value(name, dtype, shape)
    nodes.append(
        ir.Node(
            function.domain,
            function.name,
            inputs=inputs,
            attributes=[],
            outputs=[output],
            version=None,
        )
    )
    return output


def _broadcast_indices(
    nodes: list[ir.Node], scores: ir.Value
) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]:
    shape = _node(
        nodes,
        "Shape",
        [scores],
        name="scores_shape",
        dtype=ir.DataType.INT64,
        shape=[4],
    )
    zero = _const(nodes, "zero", 0, ir.DataType.INT32, [])
    one = _const(nodes, "one", 1, ir.DataType.INT32, [])

    dims = []
    for axis, dim_name in enumerate(
        ("batch_dim", "head_dim", "q_len_dim", "k_len_dim")
    ):
        axis_const = _const(nodes, f"axis_{axis}", axis, ir.DataType.INT64, [])
        dim = _node(
            nodes,
            "Gather",
            [shape, axis_const],
            name=dim_name,
            dtype=ir.DataType.INT64,
            shape=[],
            attributes={"axis": 0},
        )
        dims.append(
            _node(
                nodes,
                "Cast",
                [dim],
                name=f"{dim_name}_i32",
                dtype=ir.DataType.INT32,
                shape=[],
                attributes={"to": int(TensorProto.INT32)},
            )
        )

    ranges = []
    for dim, name in zip(
        dims, ("batch_range", "head_range", "q_range", "k_range"), strict=True
    ):
        ranges.append(
            _node(nodes, "Range", [zero, dim, one], name=name, dtype=ir.DataType.INT32)
        )

    reshape_specs = (
        ("batch", [-1, 1, 1, 1], ["B", 1, 1, 1]),
        ("head", [1, -1, 1, 1], [1, "H", 1, 1]),
        ("q_idx", [1, 1, -1, 1], [1, 1, "L", 1]),
        ("k_idx", [1, 1, 1, -1], [1, 1, 1, "S"]),
    )
    indices = []
    for range_value, (name, reshape_shape, output_shape) in zip(
        ranges, reshape_specs, strict=True
    ):
        shape_const = _const(
            nodes, f"{name}_shape", reshape_shape, ir.DataType.INT64, [4]
        )
        indices.append(
            _node(
                nodes,
                "Reshape",
                [range_value, shape_const],
                name=name,
                dtype=ir.DataType.INT32,
                shape=output_shape,
            )
        )
    return tuple(indices)  # type: ignore[return-value]


def _build_score_mod_graph(
    score_mod: ir.Function,
    mask_mod: ir.Function | None,
    score_mod_other_buffers: Sequence[ir.Value],
    mask_mod_other_buffers: Sequence[ir.Value],
) -> ir.Graph:
    nodes: list[ir.Node] = []
    scores = _tensor_value("scores", ir.DataType.FLOAT, ["B", "H", "L", "S"])
    batch, head, q_idx, k_idx = _broadcast_indices(nodes, scores)

    score_mod_result = _function_call(
        nodes,
        score_mod,
        [scores, batch, head, q_idx, k_idx, *score_mod_other_buffers],
        name="score_mod_result",
        dtype=ir.DataType.FLOAT,
        shape=["B", "H", "L", "S"],
    )

    batch_valid = _node(
        nodes,
        "Equal",
        [batch, batch],
        name="batch_index_valid",
        dtype=ir.DataType.BOOL,
        shape=["B", 1, 1, 1],
    )
    head_valid = _node(
        nodes,
        "Equal",
        [head, head],
        name="head_index_valid",
        dtype=ir.DataType.BOOL,
        shape=[1, "H", 1, 1],
    )
    index_valid = _node(
        nodes,
        "And",
        [batch_valid, head_valid],
        name="index_valid",
        dtype=ir.DataType.BOOL,
        shape=["B", "H", 1, 1],
    )

    if mask_mod is not None:
        mask = _function_call(
            nodes,
            mask_mod,
            [batch, head, q_idx, k_idx, *mask_mod_other_buffers],
            name="mask_mod_result",
            dtype=ir.DataType.BOOL,
            shape=["B", "H", "L", "S"],
        )
        condition = _node(
            nodes,
            "And",
            [mask, index_valid],
            name="combined_mask",
            dtype=ir.DataType.BOOL,
            shape=["B", "H", "L", "S"],
        )
    else:
        condition = index_valid

    neg_inf = _const(nodes, "neg_inf", float("-inf"), ir.DataType.FLOAT, [])
    output = _node(
        nodes,
        "Where",
        [condition, score_mod_result, neg_inf],
        name="modified_scores",
        dtype=ir.DataType.FLOAT,
        shape=["B", "H", "L", "S"],
    )

    return ir.Graph(
        inputs=[scores],
        outputs=[output],
        nodes=nodes,
        name="flex_attention_score_mod",
        opset_imports={"": MAIN_OPSET, "pkg.torch.__subgraph__": 1},
    )


def _as_value_sequence(values: Sequence[Any], arg_name: str) -> tuple[ir.Value, ...]:
    result = tuple(values)
    if not all(isinstance(value, ir.Value) for value in result):
        bad = [
            type(value).__name__ for value in result if not isinstance(value, ir.Value)
        ]
        raise TypeError(f"{arg_name} must contain only ONNX IR Values, got {bad}")
    return result  # type: ignore[return-value]


def flex_attention_translation(
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    score_mod: ir.Function,
    block_mask: Sequence[Any],
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: Sequence[Any] = (),
    mask_mod_other_buffers: Sequence[Any] = (),
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """Lower torch.ops.higher_order.flex_attention to ai.onnx.preview::FlexAttention.

    Version-sensitive implementation notes:
    - Verified with PyTorch 2.12.0+cu130.
    - PyTorch's dynamo exporter passes score_mod and block_mask[-1] mask_mod as
      local onnx_ir.Function objects translated from FX GraphModules.
    - This custom translation runs while torch.onnx._internal.exporter._core.current_tracer
      points at the active OpRecorder, so the custom node can be appended directly.
    """
    if kernel_options.get("OUTPUT_LOGSUMEXP") or kernel_options.get("OUTPUT_MAX"):
        raise NotImplementedError(
            "ai.onnx.preview::FlexAttention export supports only the attention output; "
            "OUTPUT_LOGSUMEXP and OUTPUT_MAX must be false."
        )
    if not isinstance(score_mod, ir.Function):
        raise TypeError(
            f"score_mod must be an ONNX IR Function, got {type(score_mod)!r}"
        )

    mask_mod = block_mask[-1] if block_mask else None
    if mask_mod is not None and not isinstance(mask_mod, ir.Function):
        mask_mod = None

    score_buffers = _as_value_sequence(
        score_mod_other_buffers, "score_mod_other_buffers"
    )
    mask_buffers = _as_value_sequence(mask_mod_other_buffers, "mask_mod_other_buffers")
    score_mod_graph = _build_score_mod_graph(
        score_mod, mask_mod, score_buffers, mask_buffers
    )

    output = ir.Value(name=None)
    node = ir.Node(
        PREVIEW_DOMAIN,
        "FlexAttention",
        inputs=[query, key, value],
        attributes=ir.convenience.convert_attributes(
            {
                "scale": float(scale),
                "softmax_precision": int(TensorProto.FLOAT),
                "score_mod": score_mod_graph,
            }
        ),
        outputs=[output],
        version=PREVIEW_OPSET,
    )

    tracer = _core.current_tracer
    if tracer is None:
        raise RuntimeError(
            "FlexAttention custom translation requires an active PyTorch ONNX OpRecorder"
        )
    tracer.nodes.append(node)

    # PyTorch's HOP returns (out, logsumexp, max_scores). The ONNX preview op has
    # one output, so lse/max are inert placeholders and are rejected above when requested.
    return output, ir.Value(name=None), ir.Value(name=None)


def export_model_proto(
    model: torch.nn.Module,
    dummy_x: torch.Tensor,
    opset: int,
) -> onnx.ModelProto:
    with torch.no_grad():
        onnx_program = torch.onnx.export(
            model,
            (dummy_x,),
            f=None,
            input_names=["x"],
            output_names=["y"],
            opset_version=opset,
            dynamo=True,
            custom_translation_table={
                torch.ops.higher_order.flex_attention: flex_attention_translation
            },
            optimize=False,
        )
    if onnx_program is None:
        raise RuntimeError(
            "torch.onnx.export(dynamo=True) did not return an ONNXProgram"
        )
    return onnx_program.model_proto


def _score_tensor_type():
    return helper.make_tensor_value_info(
        "scores",
        TensorProto.FLOAT,
        ["B", "H", "L", "S"],
    ).type


def annotate_score_mod_types(model: onnx.ModelProto) -> None:
    score_type = _score_tensor_type()

    updated = False
    for node in model.graph.node:
        if node.domain != PREVIEW_DOMAIN or node.op_type != "FlexAttention":
            continue

        score_mod = next(
            (attr.g for attr in node.attribute if attr.name == "score_mod"), None
        )
        if score_mod is None:
            continue
        if len(score_mod.input) != 1 or len(score_mod.output) != 1:
            raise ValueError(
                "FlexAttention score_mod graph must have exactly one input and one output"
            )

        score_mod.input[0].type.CopyFrom(score_type)
        score_mod.output[0].type.CopyFrom(score_type)
        updated = True

    if not updated:
        raise ValueError(
            f"No {PREVIEW_DOMAIN}::FlexAttention score_mod graph found to annotate"
        )


def verify_model(model: onnx.ModelProto) -> None:
    checker.check_model(model)

    flex_nodes = [
        node
        for node in model.graph.node
        if node.domain == PREVIEW_DOMAIN and node.op_type == "FlexAttention"
    ]
    if len(flex_nodes) != 1:
        raise ValueError(
            f"Expected exactly one {PREVIEW_DOMAIN}::FlexAttention node, found {len(flex_nodes)}"
        )

    flex_node = flex_nodes[0]
    if len(flex_node.input) != 3:
        raise ValueError(
            f"FlexAttention must have query/key/value inputs only, found {len(flex_node.input)}"
        )

    attr_names = {attr.name for attr in flex_node.attribute}
    required_attrs = {"scale", "softmax_precision", "score_mod"}
    missing_attrs = required_attrs - attr_names
    if missing_attrs:
        raise ValueError(
            f"FlexAttention node is missing attributes: {sorted(missing_attrs)}"
        )

    score_mod = next(attr.g for attr in flex_node.attribute if attr.name == "score_mod")
    score_type = _score_tensor_type()
    if len(score_mod.input) != 1 or score_mod.input[0].type != score_type:
        raise ValueError("score_mod input must be annotated as tensor(float)[B,H,L,S]")
    if len(score_mod.output) != 1 or score_mod.output[0].type != score_type:
        raise ValueError("score_mod output must be annotated as tensor(float)[B,H,L,S]")

    score_mod_ops = {node.op_type for node in score_mod.node}
    required_ops = {"Shape", "Gather", "Range", "Reshape", "Where"}
    missing_ops = required_ops - score_mod_ops
    if missing_ops:
        raise ValueError(
            f"score_mod graph is missing dynamic score_mod ops: {sorted(missing_ops)}"
        )

    input_shape = [
        dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim
    ]
    output_shape = [
        dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim
    ]
    if input_shape != output_shape:
        raise ValueError(
            f"Expected matching input/output shapes, got {input_shape} and {output_shape}"
        )

    print("ONNX checker passed")
    print(f"Found {PREVIEW_DOMAIN}::FlexAttention with score_mod")
    print(f"Input/output shape: {input_shape}")


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
        seed=args.seed,
    )
    dummy_x = torch.rand(
        args.batch, args.seq_len, args.embed_dim, dtype=torch_dtype(args.dtype)
    )
    model_proto = export_model_proto(model, dummy_x, args.opset)
    annotate_score_mod_types(model_proto)
    verify_model(model_proto)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_proto, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
