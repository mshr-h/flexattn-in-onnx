from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import (
    AttributeProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    helper,
    numpy_helper,
)

try:
    from .canonical_flexattention import CAPTURE_NAMES, PREVIEW_DOMAIN, PREVIEW_OPSET
except ImportError:
    from canonical_flexattention import CAPTURE_NAMES, PREVIEW_DOMAIN, PREVIEW_OPSET


class PreflightError(ValueError):
    pass


def _attributes(node: NodeProto) -> dict[str, AttributeProto]:
    return {attribute.name: attribute for attribute in node.attribute}


def _producer_map(graph: GraphProto) -> dict[str, NodeProto]:
    producers: dict[str, NodeProto] = {}
    for node in graph.node:
        for output in node.output:
            if output:
                if output in producers:
                    raise PreflightError(f"score_mod value {output!r} has multiple producers")
                producers[output] = node
    for initializer in graph.initializer:
        if initializer.name in producers:
            raise PreflightError(
                f"score_mod value {initializer.name!r} has multiple producers"
            )
        producers[initializer.name] = helper.make_node(
            "Constant", [], [initializer.name], value=initializer
        )
    return producers


def _constant_array(node: NodeProto) -> np.ndarray:
    if node.domain or node.op_type != "Constant":
        raise PreflightError(f"expected Constant, found {node.op_type}")
    value = _attributes(node).get("value")
    if value is None or value.type != AttributeProto.TENSOR:
        raise PreflightError("Constant must use a tensor-valued 'value' attribute")
    return numpy_helper.to_array(value.t)


def _resolve_identity(producers: dict[str, NodeProto], value: str) -> str:
    visited: set[str] = set()
    while value in producers and producers[value].op_type == "Identity":
        if value in visited:
            raise PreflightError("score_mod Identity cycle detected")
        visited.add(value)
        identity = producers[value]
        if identity.domain or len(identity.input) != 1 or len(identity.output) != 1:
            raise PreflightError("Identity must have one input and output in the ONNX domain")
        value = identity.input[0]
    return value


def _expect_producer(
    producers: dict[str, NodeProto], value: str, op_type: str
) -> NodeProto:
    value = _resolve_identity(producers, value)
    node = producers.get(value)
    if node is None:
        raise PreflightError(f"score_mod value {value!r} has no producer")
    if node.domain or node.op_type != op_type:
        raise PreflightError(
            f"score_mod value {value!r} must be produced by {op_type}, found {node.op_type}"
        )
    return node


def _int64_constant(producers: dict[str, NodeProto], value: str) -> np.ndarray:
    array = _constant_array(_expect_producer(producers, value, "Constant"))
    if array.dtype != np.int64:
        raise PreflightError("canonical score_mod constants must use int64")
    return array


def _reshape_source(
    producers: dict[str, NodeProto], value: str, expected_shape: tuple[int, ...]
) -> str:
    reshape = _expect_producer(producers, value, "Reshape")
    if len(reshape.input) != 2:
        raise PreflightError("Reshape must have data and shape inputs")
    actual_shape = tuple(
        int(item) for item in _int64_constant(producers, reshape.input[1]).reshape(-1)
    )
    if actual_shape != expected_shape:
        raise PreflightError(
            f"Reshape for {value!r} has shape {actual_shape}, expected {expected_shape}"
        )
    return _resolve_identity(producers, reshape.input[0])


def _split_absolute_position(
    producers: dict[str, NodeProto], value: str, index_shape: tuple[int, ...]
) -> tuple[str, NodeProto]:
    add = _expect_producer(producers, value, "Add")
    if len(add.input) != 2:
        raise PreflightError("absolute-position Add must have two inputs")
    sources = []
    for add_input in add.input:
        reshape = _expect_producer(producers, add_input, "Reshape")
        source = _resolve_identity(producers, reshape.input[0])
        shape = tuple(
            int(item)
            for item in _int64_constant(producers, reshape.input[1]).reshape(-1)
        )
        sources.append((source, shape))
    capture = next((source for source, shape in sources if shape == (-1, 1, 1, 1)), None)
    range_value = next((source for source, shape in sources if shape == index_shape), None)
    if capture is None or range_value is None:
        raise PreflightError(f"{value!r} must add a batch capture and a Range index")
    return capture, _expect_producer(producers, range_value, "Range")


def _validate_range(
    producers: dict[str, NodeProto], range_node: NodeProto, axis: int, score_input: str
) -> None:
    if len(range_node.input) != 3:
        raise PreflightError("Range must have start, limit, and step inputs")
    start = _int64_constant(producers, range_node.input[0])
    step = _int64_constant(producers, range_node.input[2])
    if start.shape != () or int(start) != 0 or step.shape != () or int(step) != 1:
        raise PreflightError("Range must use scalar int64 start=0 and step=1")
    gather = _expect_producer(producers, range_node.input[1], "Gather")
    if len(gather.input) != 2:
        raise PreflightError("Range limit Gather must have two inputs")
    shape = _expect_producer(producers, gather.input[0], "Shape")
    if len(shape.input) != 1 or _resolve_identity(producers, shape.input[0]) != score_input:
        raise PreflightError("Range limit must come from Shape(score_mod_input)")
    gathered_axis = _int64_constant(producers, gather.input[1])
    if gathered_axis.shape != () or int(gathered_axis) != axis:
        raise PreflightError(f"Range limit must gather score axis {axis}")


def _validate_score_mod(graph: GraphProto) -> None:
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise PreflightError("score_mod must have exactly one input and one output")
    score_input = graph.input[0].name
    score_type = graph.input[0].type.tensor_type
    output_type = graph.output[0].type.tensor_type
    if (
        score_type.elem_type != TensorProto.FLOAT
        or output_type.elem_type != TensorProto.FLOAT
        or len(score_type.shape.dim) != 4
        or len(output_type.shape.dim) != 4
    ):
        raise PreflightError("score_mod input and output must be rank-4 float32 tensors")

    allowed_ops = {
        "Shape",
        "Constant",
        "Gather",
        "Range",
        "Reshape",
        "Add",
        "Less",
        "LessOrEqual",
        "Or",
        "Where",
        "Identity",
    }
    unsupported = sorted(
        {
            f"{node.domain}::{node.op_type}" if node.domain else node.op_type
            for node in graph.node
            if node.domain or node.op_type not in allowed_ops
        }
    )
    if unsupported:
        raise PreflightError(f"score_mod contains unsupported operators: {unsupported}")

    producers = _producer_map(graph)
    where = _expect_producer(producers, graph.output[0].name, "Where")
    if (
        len(where.input) != 3
        or _resolve_identity(producers, where.input[1]) != score_input
    ):
        raise PreflightError("score_mod must return Where(allowed, scores, -inf)")
    negative_infinity = _constant_array(
        _expect_producer(producers, where.input[2], "Constant")
    )
    if negative_infinity.dtype != np.float32 or negative_infinity.shape != ():
        raise PreflightError("score_mod masked value must be a scalar float32")
    if not np.isneginf(negative_infinity):
        raise PreflightError("score_mod masked value must be -inf")

    logical_or = _expect_producer(producers, where.input[0], "Or")
    if len(logical_or.input) != 2:
        raise PreflightError("PrefixLM condition must be a two-input Or")
    condition_nodes = []
    for value in logical_or.input:
        resolved = _resolve_identity(producers, value)
        comparison = producers.get(resolved)
        if (
            comparison is None
            or comparison.domain
            or comparison.op_type not in {"Less", "LessOrEqual"}
        ):
            actual = "missing" if comparison is None else comparison.op_type
            raise PreflightError(
                f"PrefixLM Or input must be Less or LessOrEqual, found {actual}"
            )
        condition_nodes.append(comparison)
    by_type = {node.op_type: node for node in condition_nodes}
    if set(by_type) != {"Less", "LessOrEqual"}:
        raise PreflightError("PrefixLM Or must combine Less and LessOrEqual")

    prefix_less = by_type["Less"]
    causal_less_equal = by_type["LessOrEqual"]
    if len(prefix_less.input) != 2 or len(causal_less_equal.input) != 2:
        raise PreflightError("PrefixLM comparisons must have two inputs")
    k_absolute = _resolve_identity(producers, prefix_less.input[0])
    if _resolve_identity(producers, causal_less_equal.input[0]) != k_absolute:
        raise PreflightError("PrefixLM comparisons must use the same absolute key position")

    prefix_source = _reshape_source(
        producers, prefix_less.input[1], (-1, 1, 1, 1)
    )
    kv_source, key_range = _split_absolute_position(
        producers, k_absolute, (1, 1, 1, -1)
    )
    q_source, query_range = _split_absolute_position(
        producers, causal_less_equal.input[1], (1, 1, -1, 1)
    )
    if (prefix_source, q_source, kv_source) != CAPTURE_NAMES:
        raise PreflightError(
            "score_mod captures must map to prefix_len, q_start, and kv_start"
        )
    _validate_range(producers, query_range, 2, score_input)
    _validate_range(producers, key_range, 3, score_input)

    declared = {value.name for value in graph.input} | {
        initializer.name for initializer in graph.initializer
    }
    produced = set(producers)
    captures = {
        value
        for node in graph.node
        for value in node.input
        if value and value not in declared and value not in produced
    }
    if captures != set(CAPTURE_NAMES):
        raise PreflightError(
            "score_mod outer-scope captures are "
            f"{sorted(captures)}, expected {list(CAPTURE_NAMES)}"
        )

    reachable: set[int] = set()
    graph_node_ids = {id(node) for node in graph.node}

    def visit(value: str) -> None:
        node = producers.get(value)
        if node is None or id(node) in reachable:
            return
        if id(node) in graph_node_ids:
            reachable.add(id(node))
        for node_input in node.input:
            visit(node_input)

    visit(graph.output[0].name)
    if len(reachable) != len(graph.node):
        raise PreflightError(
            "score_mod contains nodes outside the canonical output DAG"
        )


def _validate_identity_prob_mod(graph: GraphProto) -> None:
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise PreflightError("prob_mod must have exactly one input and one output")
    for value in (graph.input[0], graph.output[0]):
        tensor_type = value.type.tensor_type
        if (
            tensor_type.elem_type != TensorProto.FLOAT
            or len(tensor_type.shape.dim) != 4
        ):
            raise PreflightError(
                "prob_mod input and output must be rank-4 float32 tensors"
            )

    producers: dict[str, NodeProto] = {}
    for node in graph.node:
        if (
            node.domain
            or node.op_type != "Identity"
            or len(node.input) != 1
            or len(node.output) != 1
        ):
            raise PreflightError("prob_mod must be absent or identity")
        if node.output[0] in producers:
            raise PreflightError("prob_mod contains duplicate value producers")
        producers[node.output[0]] = node

    value = graph.output[0].name
    visited: set[str] = set()
    while value != graph.input[0].name:
        node = producers.get(value)
        if node is None or value in visited:
            raise PreflightError("prob_mod must be absent or an identity chain")
        visited.add(value)
        value = node.input[0]
    if len(visited) != len(graph.node):
        raise PreflightError("prob_mod contains unused Identity nodes")


def _dimension_key(dimension: onnx.TensorShapeProto.Dimension) -> tuple[str, object]:
    if dimension.HasField("dim_value"):
        return ("value", dimension.dim_value)
    if dimension.HasField("dim_param"):
        return ("param", dimension.dim_param)
    return ("unknown", None)


def _validate_capture_declarations(model: ModelProto) -> None:
    inputs = {value.name: value for value in model.graph.input}
    initializers = {value.name: value for value in model.graph.initializer}
    query = inputs.get("Q")
    if query is None or len(query.type.tensor_type.shape.dim) != 4:
        raise PreflightError("Q must be a rank-4 graph input")
    batch_dimension = _dimension_key(query.type.tensor_type.shape.dim[0])

    for name in CAPTURE_NAMES:
        if name in initializers:
            initializer = initializers[name]
            if initializer.data_type != TensorProto.INT64 or len(initializer.dims) != 1:
                raise PreflightError(f"{name} initializer must have dtype int64 and shape [B]")
            if batch_dimension[0] == "value" and initializer.dims[0] != batch_dimension[1]:
                raise PreflightError(f"{name} initializer batch dimension does not match Q")
            continue

        value = inputs.get(name)
        if value is None:
            raise PreflightError(f"{name} must be a graph input or initializer")
        tensor_type = value.type.tensor_type
        dimensions = tensor_type.shape.dim
        if tensor_type.elem_type != TensorProto.INT64 or len(dimensions) != 1:
            raise PreflightError(f"{name} must have dtype int64 and shape [B]")
        capture_dimension = _dimension_key(dimensions[0])
        if (
            batch_dimension[0] != "unknown"
            and capture_dimension[0] != "unknown"
            and capture_dimension != batch_dimension
        ):
            raise PreflightError(f"{name} batch dimension does not match Q")


def _tensor_metadata(model: ModelProto, name: str) -> tuple[int, int] | None:
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        if value.name == name and value.type.HasField("tensor_type"):
            tensor_type = value.type.tensor_type
            return tensor_type.elem_type, len(tensor_type.shape.dim)
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return initializer.data_type, len(initializer.dims)
    return None


def validate_model(model: ModelProto) -> list[str]:
    opsets = {opset.domain: opset.version for opset in model.opset_import}
    if opsets.get(PREVIEW_DOMAIN) != PREVIEW_OPSET:
        raise PreflightError(
            f"model must import {PREVIEW_DOMAIN} opset {PREVIEW_OPSET}"
        )
    _validate_capture_declarations(model)

    top_level_nodes = [
        node
        for node in model.graph.node
        if node.domain == PREVIEW_DOMAIN and node.op_type == "FlexAttention"
    ]
    if not top_level_nodes:
        raise PreflightError("model has no top-level ai.onnx.preview::FlexAttention node")

    names = [node.name for node in top_level_nodes]
    if any(not name for name in names):
        raise PreflightError("every FlexAttention node must have a non-empty name")
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise PreflightError(f"FlexAttention node names must be unique: {duplicates}")

    for node in model.graph.node:
        for attribute in node.attribute:
            subgraphs = []
            if attribute.type == AttributeProto.GRAPH:
                subgraphs = [attribute.g]
            elif attribute.type == AttributeProto.GRAPHS:
                subgraphs = list(attribute.graphs)
            for subgraph in subgraphs:
                if any(
                    nested.domain == PREVIEW_DOMAIN
                    and nested.op_type == "FlexAttention"
                    for nested in subgraph.node
                ):
                    raise PreflightError("nested FlexAttention nodes are not supported")

    for node in top_level_nodes:
        if len(node.input) != 3 or len(node.output) != 1:
            raise PreflightError(
                f"FlexAttention node {node.name!r} must have three inputs and one output"
            )
        tensor_metadata = [
            _tensor_metadata(model, name) for name in (*node.input, node.output[0])
        ]
        if any(metadata is None for metadata in tensor_metadata):
            raise PreflightError(
                f"FlexAttention node {node.name!r} requires type information for Q, K, V, and Y"
            )
        concrete_metadata = [
            metadata for metadata in tensor_metadata if metadata is not None
        ]
        element_types = {metadata[0] for metadata in concrete_metadata}
        supported_types = {
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.BFLOAT16,
        }
        if element_types.isdisjoint(supported_types) or len(element_types) != 1:
            raise PreflightError(
                f"FlexAttention node {node.name!r} requires matching fp32, fp16, or bf16 Q/K/V/Y"
            )
        if any(metadata[1] != 4 for metadata in concrete_metadata):
            raise PreflightError(
                f"FlexAttention node {node.name!r} requires rank-4 Q, K, V, and Y"
            )
        attributes = _attributes(node)
        scale = attributes.get("scale")
        if scale is not None and (
            scale.type != AttributeProto.FLOAT or not math.isfinite(scale.f)
        ):
            raise PreflightError(
                f"FlexAttention node {node.name!r} requires a finite float scale"
            )
        softmax_precision = attributes.get("softmax_precision")
        if (
            softmax_precision is None
            or softmax_precision.type != AttributeProto.INT
            or softmax_precision.i != TensorProto.FLOAT
        ):
            raise PreflightError(
                f"FlexAttention node {node.name!r} must use float32 softmax_precision"
            )
        score_mod = attributes.get("score_mod")
        if score_mod is None or score_mod.type != AttributeProto.GRAPH:
            raise PreflightError(
                f"FlexAttention node {node.name!r} must have a score_mod graph"
            )
        _validate_score_mod(score_mod.g)
        prob_mod = attributes.get("prob_mod")
        if prob_mod is not None:
            if prob_mod.type != AttributeProto.GRAPH:
                raise PreflightError("prob_mod must be a graph")
            _validate_identity_prob_mod(prob_mod.g)

    onnx.checker.check_model(model, full_check=True)
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the canonical PrefixLM FlexAttention model contract."
    )
    parser.add_argument("model", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = validate_model(onnx.load(args.model))
    print(f"validated {len(names)} FlexAttention node(s): {', '.join(names)}")


if __name__ == "__main__":
    main()
