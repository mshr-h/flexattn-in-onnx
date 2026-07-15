from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto

try:
    from .canonical_flexattention import (
        FLEX_ATTENTION_NODE_NAME,
        PREVIEW_DOMAIN,
        ModelConfig,
        build_model,
        iter_graphs,
        prefixlm_reference,
    )
    from .expanded_attention import ExpandedConfig, build_expanded_model
    from .preflight_flexattention import validate_model
except ImportError:
    from canonical_flexattention import (
        FLEX_ATTENTION_NODE_NAME,
        PREVIEW_DOMAIN,
        ModelConfig,
        build_model,
        iter_graphs,
        prefixlm_reference,
    )
    from expanded_attention import ExpandedConfig, build_expanded_model
    from preflight_flexattention import validate_model


CASES = (
    ("causal_p0", 1, 2, 2, 4, 4, 8, 6, (0,), (0,), (0,)),
    ("full_prefix", 1, 2, 2, 4, 4, 8, 6, (4,), (0,), (0,)),
    ("per_batch_gqa", 2, 4, 2, 3, 5, 16, 6, (0, 5), (2, 0), (0, 0)),
    ("decode", 1, 4, 2, 1, 5, 8, 8, (2,), (4,), (0,)),
    ("continued_prefill", 1, 4, 2, 3, 5, 8, 8, (2,), (2,), (0,)),
    ("offset_cache", 1, 2, 1, 2, 4, 8, 8, (11,), (12,), (10,)),
    ("large_value_workspace", 1, 2, 1, 3, 5, 64, 256, (2,), (2,), (0,)),
)

CPU_FAST_PATH_CASES = (
    ("fast_causal_p0", 1, 4, 2, 512, 512, 64, 64, (0,), (0,), (0,)),
    ("fast_full_prefix", 1, 4, 2, 512, 512, 64, 64, (512,), (0,), (0,)),
    ("fast_prefix16", 1, 4, 2, 512, 512, 64, 64, (16,), (0,), (0,)),
    (
        "fast_per_batch_offsets",
        2,
        4,
        2,
        512,
        512,
        64,
        64,
        (16, 128),
        (0, 256),
        (0, 128),
    ),
    (
        "fast_continued_prefill",
        1,
        4,
        2,
        128,
        512,
        64,
        64,
        (16,),
        (384,),
        (0,),
    ),
    ("fast_value96", 1, 4, 2, 512, 512, 64, 96, (16,), (0,), (0,)),
)


def _dtype_config(dtype: str) -> tuple[int, np.dtype, float, float]:
    if dtype == "float32":
        return TensorProto.FLOAT, np.dtype(np.float32), 1e-5, 1e-4
    if dtype == "float16":
        return TensorProto.FLOAT16, np.dtype(np.float16), 5e-3, 5e-3
    import ml_dtypes

    return TensorProto.BFLOAT16, np.dtype(ml_dtypes.bfloat16), 1e-2, 1e-2


def _run_session(
    session: Any, feeds: dict[str, np.ndarray], tensor_type: int
) -> np.ndarray:
    if tensor_type != TensorProto.BFLOAT16:
        return session.run(None, feeds)[0]

    import onnxruntime as ort
    import torch

    binding = session.io_binding()
    for name, value in feeds.items():
        if value.dtype == np.dtype("int64"):
            ort_value = ort.OrtValue.ortvalue_from_numpy(value)
        else:
            ort_value = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
                value, TensorProto.BFLOAT16
            )
        binding.bind_ortvalue_input(name, ort_value)
    binding.bind_output("Y")
    session.run_with_iobinding(binding)
    return torch.from_dlpack(binding.get_outputs()[0]).float().numpy()


def _verify_invalid_inputs(
    session: object, numpy_dtype: np.dtype, tensor_type: int
) -> None:
    query = np.ones((1, 1, 1, 8), dtype=numpy_dtype)
    key = np.ones((1, 1, 1, 8), dtype=numpy_dtype)
    value = np.ones((1, 1, 1, 8), dtype=numpy_dtype)
    valid_positions = {
        "prefix_len": np.asarray([0], dtype=np.int64),
        "q_start": np.asarray([0], dtype=np.int64),
        "kv_start": np.asarray([0], dtype=np.int64),
    }
    int64_max = np.iinfo(np.int64).max
    boundary_cases = (
        {
            "Q": np.ones((1, 1, 2, 8), dtype=numpy_dtype),
            "K": key,
            "V": value,
            "prefix_len": np.asarray([1], dtype=np.int64),
            "q_start": np.asarray([int64_max - 1], dtype=np.int64),
            "kv_start": np.asarray([0], dtype=np.int64),
        },
        {
            "Q": query,
            "K": np.ones((1, 1, 2, 8), dtype=numpy_dtype),
            "V": np.ones((1, 1, 2, 8), dtype=numpy_dtype),
            "prefix_len": np.asarray([int64_max], dtype=np.int64),
            "q_start": np.asarray([int64_max - 1], dtype=np.int64),
            "kv_start": np.asarray([int64_max - 1], dtype=np.int64),
        },
    )
    for feeds in boundary_cases:
        _run_session(session, feeds, tensor_type)

    invalid_cases = (
        (
            "negative position",
            {"Q": query, "K": key, "V": value, **valid_positions}
            | {"q_start": np.asarray([-1], dtype=np.int64)},
            "must be non-negative",
        ),
        (
            "fully masked first query",
            {"Q": query, "K": key, "V": value, **valid_positions}
            | {"kv_start": np.asarray([1], dtype=np.int64)},
            "at least one allowed key",
        ),
        (
            "empty KV",
            {
                "Q": query,
                "K": np.empty((1, 1, 0, 8), dtype=numpy_dtype),
                "V": np.empty((1, 1, 0, 8), dtype=numpy_dtype),
                **valid_positions,
            },
            "dimensions must be positive",
        ),
        (
            "query absolute position overflow",
            {
                "Q": np.ones((1, 1, 2, 8), dtype=numpy_dtype),
                "K": key,
                "V": value,
                **valid_positions,
                "q_start": np.asarray(
                    [np.iinfo(np.int64).max], dtype=np.int64
                ),
            },
            "q_start plus query length exceeds int64 range",
        ),
        (
            "key absolute position overflow",
            {
                "Q": query,
                "K": np.ones((1, 1, 2, 8), dtype=numpy_dtype),
                "V": np.ones((1, 1, 2, 8), dtype=numpy_dtype),
                **valid_positions,
                "prefix_len": np.asarray(
                    [np.iinfo(np.int64).max], dtype=np.int64
                ),
                "kv_start": np.asarray(
                    [np.iinfo(np.int64).max], dtype=np.int64
                ),
            },
            "kv_start plus K/V length exceeds int64 range",
        ),
    )
    for name, feeds, expected_message in invalid_cases:
        try:
            _run_session(session, feeds, tensor_type)
        except Exception as error:
            if expected_message not in str(error):
                raise AssertionError(
                    f"{name} failed with an unexpected error: {error}"
                ) from error
        else:
            raise AssertionError(f"{name} was not rejected")


def verify_runtime(
    output_dir: Path,
    dtype: str,
    provider: str,
    ep_library: Path | None = None,
    skip_expanded: bool = False,
) -> None:
    import onnxruntime as ort

    if ep_library is not None:
        ort.register_execution_provider_library(
            provider, str(ep_library.resolve())
        )
    tensor_type, numpy_dtype, atol, rtol = _dtype_config(dtype)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"prefixlm_{dtype}.onnx"
    optimized_path = output_dir / f"prefixlm_{dtype}.optimized.onnx"
    model = build_model(ModelConfig(q_dtype=tensor_type))
    validate_model(model)
    onnx.save(model, model_path)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(optimized_path)
    options.enable_profiling = True
    options.profile_file_prefix = str(output_dir / f"prefixlm_{dtype}_profile")
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=[provider]
    )

    cases = CASES
    if provider == "CPUExecutionProvider":
        cases += CPU_FAST_PATH_CASES
    if provider == "CUDAExecutionProvider":
        cases = tuple(
            case[:7] + ((case[7] + 7) // 8 * 8,) + case[8:]
            for case in CASES
        )

    rng = np.random.default_rng(20260714)
    for case in cases:
        (
            name,
            batch,
            q_heads,
            kv_heads,
            q_length,
            kv_length,
            qk_size,
            value_size,
            prefix_len,
            q_start,
            kv_start,
        ) = case
        query = rng.normal(
            size=(batch, q_heads, q_length, qk_size)
        ).astype(numpy_dtype)
        key = rng.normal(
            size=(batch, kv_heads, kv_length, qk_size)
        ).astype(numpy_dtype)
        value = rng.normal(
            size=(batch, kv_heads, kv_length, value_size)
        ).astype(numpy_dtype)
        positions = {
            "prefix_len": np.asarray(prefix_len, dtype=np.int64),
            "q_start": np.asarray(q_start, dtype=np.int64),
            "kv_start": np.asarray(kv_start, dtype=np.int64),
        }
        feeds = {"Q": query, "K": key, "V": value, **positions}
        actual = _run_session(session, feeds, tensor_type)
        numpy_expected = prefixlm_reference(query, key, value, **positions)
        expanded_actual = None
        if not skip_expanded:
            expanded_model = build_expanded_model(
                ExpandedConfig(
                    batch,
                    q_heads,
                    kv_heads,
                    q_length,
                    kv_length,
                    qk_size,
                    value_size,
                    tensor_type,
                    cast_inputs_to_float32=(
                        tensor_type == TensorProto.BFLOAT16
                    ),
                )
            )
            expanded_options = ort.SessionOptions()
            expanded_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            expanded_session = ort.InferenceSession(
                expanded_model.SerializeToString(),
                sess_options=expanded_options,
                providers=[provider],
            )
            expanded_actual = _run_session(
                expanded_session, feeds, tensor_type
            )
        np.testing.assert_allclose(
            actual.astype(np.float32),
            numpy_expected.astype(np.float32),
            atol=atol,
            rtol=rtol,
            err_msg=f"{name}: direct versus NumPy",
        )
        if expanded_actual is not None:
            np.testing.assert_allclose(
                expanded_actual.astype(np.float32),
                numpy_expected.astype(np.float32),
                atol=atol,
                rtol=rtol,
                err_msg=f"{name}: expanded versus NumPy",
            )

    profile_path = Path(session.end_profiling())
    optimized = onnx.load(optimized_path)
    flex_nodes = [
        node
        for node in optimized.graph.node
        if node.domain == PREVIEW_DOMAIN and node.op_type == "FlexAttention"
    ]
    if len(flex_nodes) != 1 or flex_nodes[0].name != FLEX_ATTENTION_NODE_NAME:
        raise AssertionError("optimized graph did not retain the FlexAttention node")
    expanded_ops = {
        node.op_type
        for graph in iter_graphs(optimized.graph)
        for node in graph.node
        if node.op_type in {"MatMul", "Softmax"}
    }
    if expanded_ops:
        raise AssertionError(f"optimized graph contains expanded ops: {expanded_ops}")

    events = json.loads(profile_path.read_text())
    flex_events = [
        event
        for event in events
        if event.get("cat") == "Node"
        and event.get("args", {}).get("op_name") == "FlexAttention"
    ]
    if len(flex_events) != len(cases):
        raise AssertionError(
            f"expected {len(cases)} FlexAttention events, found {len(flex_events)}"
        )
    providers = {event.get("args", {}).get("provider") for event in flex_events}
    if providers != {provider}:
        raise AssertionError(f"FlexAttention ran on {providers}, expected {provider}")
    _verify_invalid_inputs(session, numpy_dtype, tensor_type)
    print(f"verified {len(cases)} {dtype} cases on {provider}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument(
        "--ep-library",
        type=Path,
        help="Plugin EP shared library to register under --provider before session creation.",
    )
    parser.add_argument(
        "--skip-expanded",
        action="store_true",
        help="Skip the primitive comparison when the selected EP lacks its dtype kernels.",
    )
    args = parser.parse_args()
    if args.output_dir:
        verify_runtime(
            args.output_dir,
            args.dtype,
            args.provider,
            args.ep_library,
            args.skip_expanded,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="prefixlm-verify-") as temporary:
            verify_runtime(
                Path(temporary),
                args.dtype,
                args.provider,
                args.ep_library,
                args.skip_expanded,
            )


if __name__ == "__main__":
    main()
