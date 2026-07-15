from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper

from prefixlm.canonical_flexattention import (
    CAPTURE_NAMES,
    ModelConfig,
    build_model,
    prefixlm_reference,
)
from prefixlm.expanded_attention import ExpandedConfig, build_expanded_model
from prefixlm.export_canonical_flexattention import export_model
from prefixlm.modifier_corpus import models as modifier_corpus_models
from prefixlm.preflight_flexattention import PreflightError, validate_model


class CanonicalFlexAttentionTest(unittest.TestCase):
    def test_shared_modifier_corpus(self) -> None:
        for case, model in modifier_corpus_models():
            with self.subTest(case=case.name):
                if case.accepted:
                    validate_model(model)
                else:
                    with self.assertRaises(PreflightError):
                        validate_model(model)

    def test_model_passes_onnx_checker_and_preflight(self) -> None:
        model = build_model()
        onnx.checker.check_model(model, full_check=True)
        self.assertEqual(validate_model(model), ["prefixlm_flex_attention"])

        flex = model.graph.node[0]
        self.assertEqual(list(flex.input), ["Q", "K", "V"])
        score_mod = next(attr.g for attr in flex.attribute if attr.name == "score_mod")
        declared = {value.name for value in score_mod.input}
        produced = {output for node in score_mod.node for output in node.output}
        captures = {
            value
            for node in score_mod.node
            for value in node.input
            if value and value not in declared and value not in produced
        }
        self.assertEqual(captures, set(CAPTURE_NAMES))

    def test_preflight_ignores_node_names(self) -> None:
        model = build_model(ModelConfig(node_name="renamed_attention"))
        score_mod = next(
            attr.g for attr in model.graph.node[0].attribute if attr.name == "score_mod"
        )
        for index, node in enumerate(score_mod.node):
            node.name = f"arbitrary_{index}"
        self.assertEqual(validate_model(model), ["renamed_attention"])

    def test_preflight_rejects_non_identity_prob_mod(self) -> None:
        model = build_model()
        flex = model.graph.node[0]
        probability_type = helper.make_tensor_type_proto(
            TensorProto.FLOAT, ["B", "Hq", "L", "S"]
        )
        prob_mod = helper.make_graph(
            [helper.make_node("Mul", ["probabilities", "probabilities"], ["out"])],
            "non_identity_prob_mod",
            [helper.make_value_info("probabilities", probability_type)],
            [helper.make_value_info("out", probability_type)],
        )
        flex.attribute.append(helper.make_attribute("prob_mod", prob_mod))
        with self.assertRaisesRegex(PreflightError, "prob_mod must be absent or identity"):
            validate_model(model)

    def test_preflight_accepts_identity_chain_prob_mod(self) -> None:
        model = build_model()
        flex = model.graph.node[0]
        probability_type = helper.make_tensor_type_proto(
            TensorProto.FLOAT, ["B", "Hq", "L", "S"]
        )
        prob_mod = helper.make_graph(
            [
                helper.make_node("Identity", ["probabilities"], ["probabilities_1"]),
                helper.make_node("Identity", ["probabilities_1"], ["probabilities_out"]),
            ],
            "identity_prob_mod",
            [helper.make_value_info("probabilities", probability_type)],
            [helper.make_value_info("probabilities_out", probability_type)],
        )
        flex.attribute.append(helper.make_attribute("prob_mod", prob_mod))
        self.assertEqual(validate_model(model), ["prefixlm_flex_attention"])

    def test_preflight_rejects_non_float32_prob_mod(self) -> None:
        model = build_model()
        flex = model.graph.node[0]
        probability_type = helper.make_tensor_type_proto(
            TensorProto.FLOAT16, ["B", "Hq", "L", "S"]
        )
        prob_mod = helper.make_graph(
            [],
            "wrong_type_prob_mod",
            [helper.make_value_info("probabilities", probability_type)],
            [helper.make_value_info("probabilities", probability_type)],
        )
        flex.attribute.append(helper.make_attribute("prob_mod", prob_mod))
        with self.assertRaisesRegex(PreflightError, "rank-4 float32"):
            validate_model(model)

    def test_preflight_accepts_identity_normalization(self) -> None:
        model = build_model()
        score_mod = next(
            attr.g for attr in model.graph.node[0].attribute if attr.name == "score_mod"
        )
        original_output = score_mod.output[0].name
        identity_output = f"{original_output}_identity"
        score_mod.node.append(
            helper.make_node("Identity", [original_output], [identity_output])
        )
        score_mod.output[0].name = identity_output
        self.assertEqual(validate_model(model), ["prefixlm_flex_attention"])

    def test_preflight_accepts_identity_inside_canonical_dag(self) -> None:
        model = build_model()
        score_mod = next(
            attr.g for attr in model.graph.node[0].attribute if attr.name == "score_mod"
        )
        add_index = next(
            index
            for index, node in enumerate(score_mod.node)
            if "k_absolute" in node.output
        )
        score_mod.node.insert(
            add_index + 1,
            helper.make_node("Identity", ["k_absolute"], ["k_absolute_identity"]),
        )
        for node in score_mod.node:
            if node.op_type in {"Less", "LessOrEqual"} and node.input[0] == "k_absolute":
                node.input[0] = "k_absolute_identity"
        self.assertEqual(validate_model(model), ["prefixlm_flex_attention"])

    def test_preflight_rejects_unused_allowed_node(self) -> None:
        model = build_model()
        score_mod = next(
            attr.g for attr in model.graph.node[0].attribute if attr.name == "score_mod"
        )
        score_mod.node.append(helper.make_node("Identity", ["scores"], ["unused"]))
        with self.assertRaisesRegex(PreflightError, "outside the canonical output DAG"):
            validate_model(model)

    def test_preflight_accepts_folded_constant_initializer(self) -> None:
        model = build_model()
        score_mod = next(
            attribute.g
            for attribute in model.graph.node[0].attribute
            if attribute.name == "score_mod"
        )
        constant = next(
            node
            for node in score_mod.node
            if list(node.output) == ["negative_infinity"]
        )
        tensor = copy.deepcopy(
            next(attribute.t for attribute in constant.attribute if attribute.name == "value")
        )
        tensor.name = "negative_infinity"
        score_mod.initializer.append(tensor)
        score_mod.node.remove(constant)
        validate_model(model)

    def test_preflight_rejects_non_rank4_score_mod(self) -> None:
        model = build_model()
        score_mod = next(
            attr.g for attr in model.graph.node[0].attribute if attr.name == "score_mod"
        )
        score_mod.input[0].type.tensor_type.shape.dim.pop()
        with self.assertRaisesRegex(PreflightError, "rank-4 float32"):
            validate_model(model)

    def test_preflight_rejects_non_float32_softmax(self) -> None:
        model = build_model()
        precision = next(
            attr for attr in model.graph.node[0].attribute
            if attr.name == "softmax_precision"
        )
        precision.i = TensorProto.FLOAT16
        with self.assertRaisesRegex(PreflightError, "float32 softmax_precision"):
            validate_model(model)

    def test_preflight_rejects_additional_mask(self) -> None:
        model = build_model()
        score_mod = next(
            attr.g for attr in model.graph.node[0].attribute if attr.name == "score_mod"
        )
        where = score_mod.node[-1]
        self.assertEqual(where.op_type, "Where")
        score_mod.node.insert(
            len(score_mod.node) - 1,
            helper.make_node("And", [where.input[0], "padding_mask"], ["combined_mask"]),
        )
        where.input[0] = "combined_mask"
        model.graph.input.append(
            helper.make_tensor_value_info(
                "padding_mask", TensorProto.BOOL, ["B", 1, "L", "S"]
            )
        )
        with self.assertRaisesRegex(PreflightError, "unsupported operators"):
            validate_model(model)

    def test_reference_supports_dynamic_batch_prefix_and_gqa(self) -> None:
        rng = np.random.default_rng(0)
        query = rng.normal(size=(2, 4, 3, 8)).astype(np.float32)
        key = rng.normal(size=(2, 2, 5, 8)).astype(np.float32)
        value = rng.normal(size=(2, 2, 5, 6)).astype(np.float32)
        prefix_len = np.asarray([0, 5], dtype=np.int64)
        q_start = np.asarray([2, 0], dtype=np.int64)
        kv_start = np.asarray([0, 0], dtype=np.int64)

        output = prefixlm_reference(
            query, key, value, prefix_len, q_start, kv_start
        )
        self.assertEqual(output.shape, (2, 4, 3, 6))
        self.assertTrue(np.isfinite(output).all())

    def test_scale_must_be_finite_at_build_and_preflight_boundaries(self) -> None:
        for scale in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "scale must be finite"):
                    build_model(ModelConfig(scale=scale))

                model = build_model()
                model.graph.node[0].attribute.append(
                    helper.make_attribute("scale", scale)
                )
                with self.assertRaisesRegex(PreflightError, "finite float scale"):
                    validate_model(model)

        self.assertEqual(
            validate_model(build_model(ModelConfig(scale=0.125))),
            ["prefixlm_flex_attention"],
        )

    def test_invalid_export_does_not_leave_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid.onnx"
            with self.assertRaisesRegex(ValueError, "scale must be finite"):
                export_model(output, ModelConfig(scale=float("nan")))
            self.assertFalse(output.exists())

    def test_reference_accepts_last_int64_position_and_rejects_overflow(self) -> None:
        query = np.ones((1, 1, 2, 2), dtype=np.float32)
        key = np.ones((1, 1, 2, 2), dtype=np.float32)
        value = np.ones((1, 1, 2, 2), dtype=np.float32)
        int64_max = np.iinfo(np.int64).max

        prefixlm_reference(
            query,
            key[:, :, :1],
            value[:, :, :1],
            np.asarray([1], dtype=np.int64),
            np.asarray([int64_max - 1], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
        )
        with self.assertRaisesRegex(ValueError, "q_start plus query length"):
            prefixlm_reference(
                query,
                key[:, :, :1],
                value[:, :, :1],
                np.asarray([1], dtype=np.int64),
                np.asarray([int64_max], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
            )

        prefixlm_reference(
            query[:, :, :1],
            key,
            value,
            np.asarray([int64_max], dtype=np.int64),
            np.asarray([int64_max - 1], dtype=np.int64),
            np.asarray([int64_max - 1], dtype=np.int64),
        )
        with self.assertRaisesRegex(ValueError, "kv_start plus K/V length"):
            prefixlm_reference(
                query[:, :, :1],
                key,
                value,
                np.asarray([int64_max], dtype=np.int64),
                np.asarray([int64_max], dtype=np.int64),
                np.asarray([int64_max], dtype=np.int64),
            )

    def test_reference_rejects_fully_masked_first_query(self) -> None:
        query = np.ones((1, 1, 1, 2), dtype=np.float32)
        key = np.ones((1, 1, 1, 2), dtype=np.float32)
        value = np.ones((1, 1, 1, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "at least one allowed key"):
            prefixlm_reference(
                query,
                key,
                value,
                np.asarray([0], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                np.asarray([1], dtype=np.int64),
            )

    def test_preflight_rejects_duplicate_flexattention_names(self) -> None:
        model = build_model()
        duplicate = copy.deepcopy(model.graph.node[0])
        duplicate.output[0] = "Y2"
        model.graph.node.append(duplicate)
        with self.assertRaisesRegex(PreflightError, "must be unique"):
            validate_model(model)

    def test_preflight_rejects_nested_flexattention(self) -> None:
        model = build_model()
        nested_flex = copy.deepcopy(model.graph.node[0])
        nested_graph = helper.make_graph(
            [nested_flex], "nested", [], []
        )
        model.graph.node.append(
            helper.make_node(
                "If",
                ["condition"],
                ["nested_output"],
                then_branch=nested_graph,
                else_branch=copy.deepcopy(nested_graph),
            )
        )
        with self.assertRaisesRegex(PreflightError, "nested FlexAttention"):
            validate_model(model)

    def test_preflight_rejects_bad_capture_type(self) -> None:
        model = build_model()
        prefix_len = next(
            value for value in model.graph.input if value.name == "prefix_len"
        )
        prefix_len.type.tensor_type.elem_type = TensorProto.INT32
        with self.assertRaisesRegex(
            PreflightError, "prefix_len must have dtype int64 and shape"
        ):
            validate_model(model)

    def test_preflight_rejects_unsupported_qkv_dtype(self) -> None:
        model = build_model()
        for value in (*model.graph.input[:3], model.graph.output[0]):
            value.type.tensor_type.elem_type = TensorProto.DOUBLE
        with self.assertRaisesRegex(PreflightError, "matching fp32, fp16, or bf16"):
            validate_model(model)

    def test_expanded_oracle_matches_numpy_reference_with_gqa(self) -> None:
        from onnx.reference import ReferenceEvaluator

        config = ExpandedConfig(2, 4, 2, 3, 5, 8, 6)
        rng = np.random.default_rng(7)
        feeds = {
            "Q": rng.normal(size=(2, 4, 3, 8)).astype(np.float32),
            "K": rng.normal(size=(2, 2, 5, 8)).astype(np.float32),
            "V": rng.normal(size=(2, 2, 5, 6)).astype(np.float32),
            "prefix_len": np.asarray([0, 5], dtype=np.int64),
            "q_start": np.asarray([2, 0], dtype=np.int64),
            "kv_start": np.asarray([0, 0], dtype=np.int64),
        }
        actual = ReferenceEvaluator(build_expanded_model(config)).run(None, feeds)[0]
        expected = prefixlm_reference(
            feeds["Q"],
            feeds["K"],
            feeds["V"],
            feeds["prefix_len"],
            feeds["q_start"],
            feeds["kv_start"],
        )
        np.testing.assert_allclose(
            actual,
            expected,
            atol=1e-5,
            rtol=1e-4,
        )

if __name__ == "__main__":
    unittest.main()
