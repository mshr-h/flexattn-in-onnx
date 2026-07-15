from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import onnx
from onnx import GraphProto, ModelProto, NodeProto, TensorProto, helper, numpy_helper


PREVIEW_DOMAIN = "ai.onnx.preview"
PREVIEW_OPSET = 1
DEFAULT_OPSET = 18
FLEX_ATTENTION_NODE_NAME = "prefixlm_flex_attention"
CAPTURE_NAMES = ("prefix_len", "q_start", "kv_start")


@dataclass(frozen=True)
class ModelConfig:
    q_dtype: int = TensorProto.FLOAT
    scale: float | None = None
    main_opset: int = DEFAULT_OPSET
    node_name: str = FLEX_ATTENTION_NODE_NAME


def _constant(name: str, value: np.ndarray | np.generic | int | float) -> NodeProto:
    array = np.asarray(value)
    return helper.make_node(
        "Constant",
        [],
        [name],
        value=numpy_helper.from_array(array, name=f"{name}_value"),
    )


def build_prefixlm_score_mod() -> GraphProto:
    score_type = helper.make_tensor_type_proto(
        TensorProto.FLOAT, ["B", "Hq", "L", "S"]
    )
    nodes = [
        helper.make_node("Shape", ["scores"], ["score_shape"]),
        _constant("q_axis", np.int64(2)),
        _constant("k_axis", np.int64(3)),
        helper.make_node("Gather", ["score_shape", "q_axis"], ["q_limit"], axis=0),
        helper.make_node("Gather", ["score_shape", "k_axis"], ["k_limit"], axis=0),
        _constant("range_start", np.int64(0)),
        _constant("range_step", np.int64(1)),
        helper.make_node(
            "Range", ["range_start", "q_limit", "range_step"], ["q_index"]
        ),
        helper.make_node(
            "Range", ["range_start", "k_limit", "range_step"], ["k_index"]
        ),
        _constant("q_index_shape", np.asarray([1, 1, -1, 1], dtype=np.int64)),
        _constant("k_index_shape", np.asarray([1, 1, 1, -1], dtype=np.int64)),
        _constant("batch_shape", np.asarray([-1, 1, 1, 1], dtype=np.int64)),
        helper.make_node("Reshape", ["q_index", "q_index_shape"], ["q_index_4d"]),
        helper.make_node("Reshape", ["k_index", "k_index_shape"], ["k_index_4d"]),
        helper.make_node("Reshape", ["q_start", "batch_shape"], ["q_start_4d"]),
        helper.make_node("Reshape", ["kv_start", "batch_shape"], ["kv_start_4d"]),
        helper.make_node("Reshape", ["prefix_len", "batch_shape"], ["prefix_len_4d"]),
        helper.make_node("Add", ["q_start_4d", "q_index_4d"], ["q_absolute"]),
        helper.make_node("Add", ["kv_start_4d", "k_index_4d"], ["k_absolute"]),
        helper.make_node("Less", ["k_absolute", "prefix_len_4d"], ["in_prefix"]),
        helper.make_node("LessOrEqual", ["k_absolute", "q_absolute"], ["causal"]),
        helper.make_node("Or", ["in_prefix", "causal"], ["allowed"]),
        _constant("negative_infinity", np.float32(-np.inf)),
        helper.make_node(
            "Where", ["allowed", "scores", "negative_infinity"], ["scores_out"]
        ),
    ]
    return helper.make_graph(
        nodes,
        "prefixlm_score_mod",
        [helper.make_value_info("scores", score_type)],
        [helper.make_value_info("scores_out", score_type)],
    )


def build_model(config: ModelConfig = ModelConfig()) -> ModelProto:
    if not config.node_name:
        raise ValueError("FlexAttention node name must not be empty")
    if config.scale is not None and not math.isfinite(config.scale):
        raise ValueError("scale must be finite")

    qkv_shape = ["B", "Hq", "L", "Dqk"]
    kv_shape = ["B", "Hkv", "S", "Dqk"]
    value_shape = ["B", "Hkv", "S", "Dv"]
    output_shape = ["B", "Hq", "L", "Dv"]
    batch_shape = ["B"]

    flex_attributes: dict[str, object] = {
        "softmax_precision": TensorProto.FLOAT,
        "score_mod": build_prefixlm_score_mod(),
    }
    if config.scale is not None:
        flex_attributes["scale"] = float(config.scale)

    flex_attention = helper.make_node(
        "FlexAttention",
        ["Q", "K", "V"],
        ["Y"],
        name=config.node_name,
        domain=PREVIEW_DOMAIN,
        **flex_attributes,
    )
    graph = helper.make_graph(
        [flex_attention],
        "prefixlm_flexattention",
        [
            helper.make_tensor_value_info("Q", config.q_dtype, qkv_shape),
            helper.make_tensor_value_info("K", config.q_dtype, kv_shape),
            helper.make_tensor_value_info("V", config.q_dtype, value_shape),
            helper.make_tensor_value_info("prefix_len", TensorProto.INT64, batch_shape),
            helper.make_tensor_value_info("q_start", TensorProto.INT64, batch_shape),
            helper.make_tensor_value_info("kv_start", TensorProto.INT64, batch_shape),
        ],
        [helper.make_tensor_value_info("Y", config.q_dtype, output_shape)],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", config.main_opset),
            helper.make_opsetid(PREVIEW_DOMAIN, PREVIEW_OPSET),
        ],
        producer_name="flexattn-in-onnx",
    )
    model.ir_version = min(onnx.IR_VERSION, 10)
    return model


def prefixlm_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    prefix_len: np.ndarray,
    q_start: np.ndarray,
    kv_start: np.ndarray,
    scale: float | None = None,
) -> np.ndarray:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Q, K, and V must be rank-4 tensors")
    batch, q_heads, q_length, qk_head_size = query.shape
    if key.shape[0] != batch or value.shape[0] != batch:
        raise ValueError("Q, K, and V batch dimensions must match")
    kv_heads, kv_length = key.shape[1:3]
    if key.shape != (batch, kv_heads, kv_length, qk_head_size):
        raise ValueError("Q and K head sizes must match")
    if value.shape[:3] != (batch, kv_heads, kv_length):
        raise ValueError("K and V head and sequence dimensions must match")
    if kv_heads == 0 or q_heads % kv_heads != 0:
        raise ValueError("Q heads must be a multiple of KV heads")
    for name, positions in zip(CAPTURE_NAMES, (prefix_len, q_start, kv_start)):
        if positions.dtype != np.int64 or positions.shape != (batch,):
            raise ValueError(f"{name} must have dtype int64 and shape [B]")
        if np.any(positions < 0):
            raise ValueError(f"{name} must be non-negative")
    if kv_length == 0:
        raise ValueError("K/V sequence length must be positive")
    int64_max = np.iinfo(np.int64).max
    if q_length and np.any(q_start > int64_max - (q_length - 1)):
        raise ValueError("q_start plus query length exceeds int64 range")
    if np.any(kv_start > int64_max - (kv_length - 1)):
        raise ValueError("kv_start plus K/V length exceeds int64 range")
    if np.any((kv_start >= prefix_len) & (kv_start > q_start)):
        raise ValueError("the first query in each batch must have at least one allowed key")

    group_size = q_heads // kv_heads
    aligned_key = np.repeat(key, group_size, axis=1)
    aligned_value = np.repeat(value, group_size, axis=1)
    effective_scale = scale if scale is not None else qk_head_size**-0.5
    if not math.isfinite(effective_scale):
        raise ValueError("scale must be finite")
    scores = np.matmul(
        query.astype(np.float32), aligned_key.astype(np.float32).swapaxes(-1, -2)
    )
    scores *= np.float32(effective_scale)

    query_positions = q_start[:, None] + np.arange(q_length, dtype=np.int64)[None, :]
    key_positions = kv_start[:, None] + np.arange(kv_length, dtype=np.int64)[None, :]
    allowed = (key_positions[:, None, :] < prefix_len[:, None, None]) | (
        key_positions[:, None, :] <= query_positions[:, :, None]
    )
    scores = np.where(allowed[:, None, :, :], scores, -np.inf)
    row_max = scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores - row_max)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    output = np.matmul(probabilities, aligned_value.astype(np.float32))
    return output.astype(query.dtype)


def iter_graphs(graph: GraphProto) -> Iterable[GraphProto]:
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from iter_graphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from iter_graphs(subgraph)
