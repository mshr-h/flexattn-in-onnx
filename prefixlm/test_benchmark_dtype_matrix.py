from __future__ import annotations

import unittest

from onnx import TensorProto

from prefixlm.benchmark_dtype_matrix import (
    _append_completeness_failures,
    _benchmark_matrix,
    _matrix_parser,
)
from prefixlm.benchmark_runtime import (
    CPU_PROVIDER,
    dtype_spec,
    expanded_label,
    score_matrix_bytes,
)
from prefixlm.cuda_allocation import ActiveAllocationTracker
from prefixlm.expanded_attention import ExpandedConfig, build_expanded_model


class BenchmarkDTypeMatrixTest(unittest.TestCase):
    def test_cpu_only_matrix_does_not_require_cuda_library(self) -> None:
        args = _matrix_parser().parse_args(["--providers", "cpu"])
        self.assertIsNone(args.ep_library)
        self.assertEqual(
            _benchmark_matrix(args.providers),
            ((CPU_PROVIDER, ("float32", "float16", "bfloat16"), ("end_to_end",)),),
        )

    def test_dtype_specs_match_accuracy_contract(self) -> None:
        self.assertEqual(dtype_spec("float32").tensor_type, TensorProto.FLOAT)
        self.assertEqual(dtype_spec("float16").atol, 5e-3)
        self.assertEqual(dtype_spec("bfloat16").rtol, 1e-2)
        with self.assertRaisesRegex(ValueError, "unsupported dtype"):
            dtype_spec("float64")

    def test_cpu_bfloat16_expanded_casts_inputs_before_gqa(self) -> None:
        model = build_expanded_model(
            ExpandedConfig(
                1,
                4,
                2,
                8,
                8,
                64,
                64,
                TensorProto.BFLOAT16,
                cast_inputs_to_float32=True,
            )
        )
        input_casts = {
            node.input[0]: node.output[0]
            for node in model.graph.node
            if node.op_type == "Cast" and node.input[0] in {"Q", "K", "V"}
        }
        self.assertEqual(
            input_casts,
            {"Q": "Q_float", "K": "K_float", "V": "V_float_input"},
        )
        unsqueeze_inputs = {
            node.input[0]
            for node in model.graph.node
            if node.op_type == "Unsqueeze"
        }
        self.assertIn("K_float", unsqueeze_inputs)
        self.assertIn("V_float_input", unsqueeze_inputs)
        first_matmul = next(
            node for node in model.graph.node if node.op_type == "MatMul"
        )
        self.assertEqual(first_matmul.input[0], "Q_float")
        self.assertEqual(
            expanded_label(CPU_PROVIDER, dtype_spec("bfloat16")),
            "expanded_fp32_cast",
        )

    def test_cpu_memory_gate_checks_every_long_sequence(self) -> None:
        args = _matrix_parser().parse_args(
            ["--providers", "cpu", "--seq-lens", "2048", "8192"]
        )
        memory_results = []
        for seq_len in args.seq_lens:
            score_bytes = score_matrix_bytes(1, 4, seq_len, seq_len)
            for dtype in ("float32", "float16", "bfloat16"):
                expanded_name = (
                    "expanded_fp32_cast"
                    if dtype == "bfloat16"
                    else "expanded"
                )
                direct_delta = (
                    score_bytes
                    if seq_len == 2048 and dtype == "float32"
                    else 0
                )
                memory_results.extend(
                    [
                        {
                            "provider": CPU_PROVIDER,
                            "dtype": dtype,
                            "seq_len": seq_len,
                            "implementation": "direct",
                            "peak_memory_incremental_bytes": direct_delta,
                        },
                        {
                            "provider": CPU_PROVIDER,
                            "dtype": dtype,
                            "seq_len": seq_len,
                            "implementation": expanded_name,
                            "peak_memory_incremental_bytes": score_bytes,
                        },
                    ]
                )

        failures: list[str] = []
        _append_completeness_failures(
            args,
            assignments=[{}] * 3,
            latency_results=[{}] * 12,
            memory_results=memory_results,
            failures=failures,
        )
        self.assertEqual(
            failures,
            [
                "CPU float32 S=2048 direct peak delta "
                "67108864 reached score size 67108864"
            ],
        )
        failed_row = next(
            row
            for row in memory_results
            if row["provider"] == CPU_PROVIDER
            and row["dtype"] == "float32"
            and row["seq_len"] == 2048
            and row["implementation"] == "direct"
        )
        self.assertFalse(failed_row["score_buffer_gate_passed"])
        self.assertEqual(
            failed_row["score_buffer_gate_rule"],
            "incremental_peak_below_float32_score_buffer",
        )

    def test_active_allocation_tracker_records_peak_and_reset(self) -> None:
        tracker = ActiveAllocationTracker()
        tracker.allocate(1, 10)
        tracker.allocate(2, 20)
        self.assertEqual(tracker.active_bytes, 30)
        self.assertEqual(tracker.peak_active_bytes, 30)

        tracker.free(1)
        tracker.reset_peak()
        tracker.allocate(3, 5)
        self.assertEqual(tracker.active_bytes, 25)
        self.assertEqual(tracker.peak_active_bytes, 25)

        tracker.free(2)
        tracker.free(3)
        self.assertEqual(tracker.active_bytes, 0)
        with self.assertRaisesRegex(ValueError, "already active"):
            tracker.allocate(4, 1)
            tracker.allocate(4, 1)


if __name__ == "__main__":
    unittest.main()
