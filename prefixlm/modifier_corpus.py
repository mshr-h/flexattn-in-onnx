from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

try:
    from .canonical_flexattention import ModelConfig, build_model
except ImportError:
    from canonical_flexattention import ModelConfig, build_model


@dataclass(frozen=True)
class ModifierCorpusCase:
    name: str
    accepted: bool
    mutate: Callable[[onnx.ModelProto], None]


def _score_mod(model: onnx.ModelProto) -> onnx.GraphProto:
    return next(
        attribute.g
        for attribute in model.graph.node[0].attribute
        if attribute.name == "score_mod"
    )


def _arbitrary_node_names(model: onnx.ModelProto) -> None:
    model.graph.node[0].name = "renamed_attention"
    for index, node in enumerate(_score_mod(model).node):
        node.name = f"arbitrary_{index}"


def _identity_normalization(model: onnx.ModelProto) -> None:
    graph = _score_mod(model)
    output = graph.output[0].name
    graph.node.append(helper.make_node("Identity", [output], [f"{output}_identity"]))
    graph.output[0].name = f"{output}_identity"


def _initializer_constant(model: onnx.ModelProto) -> None:
    graph = _score_mod(model)
    constant = next(
        node for node in graph.node if list(node.output) == ["negative_infinity"]
    )
    tensor = copy.deepcopy(
        next(attribute.t for attribute in constant.attribute if attribute.name == "value")
    )
    tensor.name = "negative_infinity"
    graph.initializer.append(tensor)
    graph.node.remove(constant)


def _identity_prob_mod(model: onnx.ModelProto) -> None:
    tensor_type = helper.make_tensor_type_proto(
        TensorProto.FLOAT, ["B", "Hq", "L", "S"]
    )
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["probabilities"], ["probabilities_1"]),
            helper.make_node("Identity", ["probabilities_1"], ["probabilities_out"]),
        ],
        "identity_prob_mod",
        [helper.make_value_info("probabilities", tensor_type)],
        [helper.make_value_info("probabilities_out", tensor_type)],
    )
    model.graph.node[0].attribute.append(helper.make_attribute("prob_mod", graph))


def _commutative_input_order(model: onnx.ModelProto) -> None:
    for node in _score_mod(model).node:
        if node.op_type in {"Add", "Or"}:
            node.input.reverse()


def _finite_scale(model: onnx.ModelProto) -> None:
    model.graph.node[0].attribute.append(helper.make_attribute("scale", 0.125))


def _non_finite_scale(model: onnx.ModelProto) -> None:
    model.graph.node[0].attribute.append(helper.make_attribute("scale", float("nan")))


def _additional_mask(model: onnx.ModelProto) -> None:
    graph = _score_mod(model)
    where = next(node for node in graph.node if node.op_type == "Where")
    graph.node.insert(
        len(graph.node) - 1,
        helper.make_node("And", [where.input[0], "padding_mask"], ["combined_mask"]),
    )
    where.input[0] = "combined_mask"


def _unused_node(model: onnx.ModelProto) -> None:
    _score_mod(model).node.append(
        helper.make_node("Identity", ["scores"], ["unused"])
    )


def _wrong_score_rank(model: onnx.ModelProto) -> None:
    _score_mod(model).input[0].type.tensor_type.shape.dim.pop()


def _wrong_score_type(model: onnx.ModelProto) -> None:
    _score_mod(model).output[0].type.tensor_type.elem_type = TensorProto.FLOAT16


def _wrong_capture_type(model: onnx.ModelProto) -> None:
    model.graph.input[3].type.tensor_type.elem_type = TensorProto.INT32
    _score_mod(model).value_info.append(
        helper.make_tensor_value_info("prefix_len", TensorProto.INT32, ["B"])
    )


def _wrong_capture_rank(model: onnx.ModelProto) -> None:
    capture = model.graph.input[3]
    capture.type.tensor_type.shape.dim.add().dim_value = 1
    _score_mod(model).value_info.append(
        helper.make_tensor_value_info("prefix_len", TensorProto.INT64, ["B", 1])
    )


def _non_identity_prob_mod(model: onnx.ModelProto) -> None:
    tensor_type = helper.make_tensor_type_proto(
        TensorProto.FLOAT, ["B", "Hq", "L", "S"]
    )
    graph = helper.make_graph(
        [helper.make_node("Mul", ["probabilities", "probabilities"], ["out"])],
        "non_identity_prob_mod",
        [helper.make_value_info("probabilities", tensor_type)],
        [helper.make_value_info("out", tensor_type)],
    )
    model.graph.node[0].attribute.append(helper.make_attribute("prob_mod", graph))


def _nested_node(model: onnx.ModelProto) -> None:
    nested_flex = copy.deepcopy(model.graph.node[0])
    nested_graph = helper.make_graph(
        [nested_flex],
        "nested",
        [],
        [copy.deepcopy(model.graph.output[0])],
    )
    condition = numpy_helper.from_array(np.asarray(True), name="condition")
    model.graph.initializer.append(condition)
    model.graph.node.append(
        helper.make_node(
            "If",
            ["condition"],
            ["nested_output"],
            then_branch=nested_graph,
            else_branch=copy.deepcopy(nested_graph),
        )
    )


def _duplicate_node_name(model: onnx.ModelProto) -> None:
    duplicate = copy.deepcopy(model.graph.node[0])
    duplicate.output[0] = "Y2"
    model.graph.node.append(duplicate)


def _duplicate_producer(model: onnx.ModelProto) -> None:
    _score_mod(model).node.append(
        helper.make_node("Identity", ["scores"], ["q_absolute"])
    )


CASES = (
    ModifierCorpusCase("arbitrary_node_names", True, _arbitrary_node_names),
    ModifierCorpusCase("identity_normalization", True, _identity_normalization),
    ModifierCorpusCase("initializer_constant", True, _initializer_constant),
    ModifierCorpusCase("identity_prob_mod", True, _identity_prob_mod),
    ModifierCorpusCase("commutative_input_order", True, _commutative_input_order),
    ModifierCorpusCase("finite_scale", True, _finite_scale),
    ModifierCorpusCase("non_finite_scale", False, _non_finite_scale),
    ModifierCorpusCase("additional_mask", False, _additional_mask),
    ModifierCorpusCase("unused_node", False, _unused_node),
    ModifierCorpusCase("wrong_score_rank", False, _wrong_score_rank),
    ModifierCorpusCase("wrong_score_type", False, _wrong_score_type),
    ModifierCorpusCase("wrong_capture_type", False, _wrong_capture_type),
    ModifierCorpusCase("wrong_capture_rank", False, _wrong_capture_rank),
    ModifierCorpusCase("non_identity_prob_mod", False, _non_identity_prob_mod),
    ModifierCorpusCase("nested_node", False, _nested_node),
    ModifierCorpusCase("duplicate_node_name", False, _duplicate_node_name),
    ModifierCorpusCase("duplicate_producer", False, _duplicate_producer),
)


def models() -> Iterator[tuple[ModifierCorpusCase, onnx.ModelProto]]:
    for case in CASES:
        model = build_model(ModelConfig(node_name="corpus_flex_attention"))
        case.mutate(model)
        yield case, model
