from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort


def numpy_dtype(onnx_type: str) -> np.dtype:
    if onnx_type == "tensor(float16)":
        return np.dtype(np.float16)
    if onnx_type == "tensor(float)":
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported input type: {onnx_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx-path", type=str, default="multihead_attention_flexattn.onnx"
    )
    parser.add_argument("--enable-profiling", action="store_true")
    args = parser.parse_args()

    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = args.enable_profiling

    session = ort.InferenceSession(
        args.onnx_path, providers=["CPUExecutionProvider"], sess_options=sess_options
    )
    input_info = session.get_inputs()[0]

    shape = [int(dim) for dim in input_info.shape]
    dtype = numpy_dtype(input_info.type)
    x = np.random.default_rng(0).standard_normal(shape).astype(dtype)

    y = session.run(None, {input_info.name: x})[0]

    prof_file = session.end_profiling()
    print(prof_file)

    print(f"ONNX Runtime: {ort.__file__}")
    print(f"providers: {session.get_providers()}")
    print(f"input:  {input_info.name} shape={x.shape} dtype={x.dtype}")
    print(f"output: shape={y.shape} dtype={y.dtype} finite={np.isfinite(y).all()}")


if __name__ == "__main__":
    main()
