from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, helper, numpy_helper


@dataclass(frozen=True)
class ExpandedConfig:
    batch: int
    query_heads: int
    kv_heads: int
    query_length: int
    kv_length: int
    qk_head_size: int
    value_head_size: int
    dtype: int = TensorProto.FLOAT
    scale: float | None = None
    cast_inputs_to_float32: bool = False


def _constant(name: str, value: np.ndarray | np.generic) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        [],
        [name],
        value=numpy_helper.from_array(np.asarray(value), name=f"{name}_value"),
    )


def build_expanded_model(config: ExpandedConfig) -> ModelProto:
    if config.query_heads % config.kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    supported_types = (
        TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16
    )
    if config.dtype not in supported_types:
        raise ValueError("expanded oracle supports float32, float16, and bfloat16")
    group_size = config.query_heads // config.kv_heads
    float_compute = config.dtype == TensorProto.FLOAT or config.cast_inputs_to_float32
    scale = config.scale if config.scale is not None else config.qk_head_size**-0.5

    query_value = "Q"
    key_value = "K"
    value_value = "V"
    nodes = []
    if config.cast_inputs_to_float32:
        query_value = "Q_float"
        key_value = "K_float"
        value_value = "V_float_input"
        nodes.extend(
            [
                helper.make_node("Cast", ["Q"], [query_value], to=TensorProto.FLOAT),
                helper.make_node("Cast", ["K"], [key_value], to=TensorProto.FLOAT),
                helper.make_node("Cast", ["V"], [value_value], to=TensorProto.FLOAT),
            ]
        )

    nodes.extend(
        [
            _constant("gqa_axes", np.asarray([2], dtype=np.int64)),
            _constant(
                "key_expand_shape",
                np.asarray(
                    [
                        config.batch,
                        config.kv_heads,
                        group_size,
                        config.kv_length,
                        config.qk_head_size,
                    ],
                    dtype=np.int64,
                ),
            ),
            _constant(
                "value_expand_shape",
                np.asarray(
                    [
                        config.batch,
                        config.kv_heads,
                        group_size,
                        config.kv_length,
                        config.value_head_size,
                    ],
                    dtype=np.int64,
                ),
            ),
            _constant(
                "key_gqa_shape",
                np.asarray(
                    [
                        config.batch,
                        config.query_heads,
                        config.kv_length,
                        config.qk_head_size,
                    ],
                    dtype=np.int64,
                ),
            ),
            _constant(
                "value_gqa_shape",
                np.asarray(
                    [
                        config.batch,
                        config.query_heads,
                        config.kv_length,
                        config.value_head_size,
                    ],
                    dtype=np.int64,
                ),
            ),
            helper.make_node(
                "Unsqueeze", [key_value, "gqa_axes"], ["key_group_axis"]
            ),
            helper.make_node(
                "Expand", ["key_group_axis", "key_expand_shape"], ["key_grouped"]
            ),
            helper.make_node(
                "Reshape", ["key_grouped", "key_gqa_shape"], ["key_gqa"]
            ),
            helper.make_node(
                "Unsqueeze", [value_value, "gqa_axes"], ["value_group_axis"]
            ),
            helper.make_node(
                "Expand",
                ["value_group_axis", "value_expand_shape"],
                ["value_grouped"],
            ),
            helper.make_node(
                "Reshape", ["value_grouped", "value_gqa_shape"], ["value_gqa"]
            ),
            helper.make_node(
                "Transpose",
                ["key_gqa"],
                ["key_transposed"],
                perm=[0, 1, 3, 2],
            ),
            helper.make_node(
                "MatMul",
                [query_value, "key_transposed"],
                ["scores_native"],
            ),
        ]
    )
    score_value = "scores_native"
    value_value = "value_gqa"
    if not float_compute:
        nodes.extend(
            [
                helper.make_node("Cast", [score_value], ["scores_float"], to=TensorProto.FLOAT),
                helper.make_node("Cast", [value_value], ["value_float"], to=TensorProto.FLOAT),
            ]
        )
        score_value = "scores_float"
        value_value = "value_float"
    nodes.extend(
        [
            _constant("scale", np.float32(scale)),
            helper.make_node("Mul", [score_value, "scale"], ["scaled_scores"]),
            _constant("q_index", np.arange(config.query_length, dtype=np.int64).reshape(1, -1, 1)),
            _constant("k_index", np.arange(config.kv_length, dtype=np.int64).reshape(1, 1, -1)),
            _constant("batch_position_shape", np.asarray([-1, 1, 1], dtype=np.int64)),
            helper.make_node("Reshape", ["prefix_len", "batch_position_shape"], ["prefix_3d"]),
            helper.make_node("Reshape", ["q_start", "batch_position_shape"], ["q_start_3d"]),
            helper.make_node("Reshape", ["kv_start", "batch_position_shape"], ["kv_start_3d"]),
            helper.make_node("Add", ["q_start_3d", "q_index"], ["q_absolute"]),
            helper.make_node("Add", ["kv_start_3d", "k_index"], ["k_absolute"]),
            helper.make_node("Less", ["k_absolute", "prefix_3d"], ["prefix_allowed"]),
            helper.make_node("LessOrEqual", ["k_absolute", "q_absolute"], ["causal_allowed"]),
            helper.make_node("Or", ["prefix_allowed", "causal_allowed"], ["allowed_3d"]),
            _constant("mask_axis", np.asarray([1], dtype=np.int64)),
            helper.make_node("Unsqueeze", ["allowed_3d", "mask_axis"], ["allowed"]),
            _constant("negative_infinity", np.float32(-np.inf)),
            helper.make_node(
                "Where",
                ["allowed", "scaled_scores", "negative_infinity"],
                ["masked_scores"],
            ),
            helper.make_node("Softmax", ["masked_scores"], ["probabilities"], axis=-1),
            helper.make_node("MatMul", ["probabilities", value_value], ["output_float"]),
        ]
    )
    output_value = "output_float"
    if config.dtype != TensorProto.FLOAT:
        nodes.append(
            helper.make_node("Cast", [output_value], ["Y"], to=config.dtype)
        )
        output_value = "Y"

    q_shape = [config.batch, config.query_heads, config.query_length, config.qk_head_size]
    k_shape = [config.batch, config.kv_heads, config.kv_length, config.qk_head_size]
    v_shape = [config.batch, config.kv_heads, config.kv_length, config.value_head_size]
    graph = helper.make_graph(
        nodes,
        "expanded_prefixlm_attention",
        [
            helper.make_tensor_value_info("Q", config.dtype, q_shape),
            helper.make_tensor_value_info("K", config.dtype, k_shape),
            helper.make_tensor_value_info("V", config.dtype, v_shape),
            helper.make_tensor_value_info("prefix_len", TensorProto.INT64, [config.batch]),
            helper.make_tensor_value_info("q_start", TensorProto.INT64, [config.batch]),
            helper.make_tensor_value_info("kv_start", TensorProto.INT64, [config.batch]),
        ],
        [
            helper.make_tensor_value_info(
                output_value,
                config.dtype,
                [
                    config.batch,
                    config.query_heads,
                    config.query_length,
                    config.value_head_size,
                ],
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(onnx.IR_VERSION, 10)
    onnx.checker.check_model(model, full_check=True)
    return model
