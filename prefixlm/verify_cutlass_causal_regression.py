from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def build_causal_model() -> onnx.ModelProto:
    head_size = 64
    shape = [1, 1, 4, head_size]
    node = helper.make_node(
        "Attention",
        ["Q", "K", "V", "", "", "", "nonpad_kv_seqlen"],
        ["Y"],
        name="causal_attention_regression",
        is_causal=1,
    )
    graph = helper.make_graph(
        [node],
        "cutlass_causal_regression",
        [
            helper.make_tensor_value_info("Q", TensorProto.FLOAT16, shape),
            helper.make_tensor_value_info("K", TensorProto.FLOAT16, shape),
            helper.make_tensor_value_info("V", TensorProto.FLOAT16, shape),
            helper.make_tensor_value_info(
                "nonpad_kv_seqlen", TensorProto.INT64, [1]
            ),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, shape)],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 24)],
        producer_name="prefixlm-cutlass-causal-regression",
    )
    model.ir_version = 13
    onnx.checker.check_model(model)
    return model


def verify(ep_library: Path, output_dir: Path) -> None:
    # This matches ORT's native regression setup: head_size=64 remains eligible
    # for memory-efficient attention while this switch excludes FlashAttention.
    os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"

    import onnxruntime as ort

    ort.register_execution_provider_library(
        "CUDAExecutionProvider", str(ep_library.resolve())
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "cutlass_causal_regression.onnx"
    onnx.save(build_causal_model(), model_path)

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(output_dir / "cutlass_causal_profile")
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CUDAExecutionProvider"],
    )

    shape = (1, 1, 4, 64)
    query = np.ones(shape, dtype=np.float16)
    key = np.ones(shape, dtype=np.float16)
    value = np.repeat(
        np.asarray([1.0, 3.0, 7.0, 9.0], dtype=np.float16), 64
    ).reshape(shape)
    actual = session.run(
        None,
        {
            "Q": query,
            "K": key,
            "V": value,
            "nonpad_kv_seqlen": np.asarray([2], dtype=np.int64),
        },
    )[0]
    expected = np.repeat(
        np.asarray([0.0, 0.0, 1.0, 2.0], dtype=np.float32), 64
    ).reshape(shape)
    np.testing.assert_allclose(
        actual.astype(np.float32), expected, atol=2e-2, rtol=0
    )

    events = json.loads(Path(session.end_profiling()).read_text())
    attention_events = [
        event
        for event in events
        if event.get("cat") == "Node"
        and event.get("args", {}).get("op_name") == "Attention"
    ]
    if len(attention_events) != 1:
        raise AssertionError(
            f"expected one Attention event, found {len(attention_events)}"
        )
    provider = attention_events[0].get("args", {}).get("provider")
    if provider != "CUDAExecutionProvider":
        raise AssertionError(f"Attention ran on {provider}")
    print("verified causal CUTLASS memory-efficient attention regression")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        verify(args.ep_library, args.output_dir)
    else:
        with tempfile.TemporaryDirectory(
            prefix="cutlass-causal-regression-"
        ) as temporary:
            verify(args.ep_library, Path(temporary))


if __name__ == "__main__":
    main()
