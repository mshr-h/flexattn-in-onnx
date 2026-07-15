from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from onnx import TensorProto

try:
    from .canonical_flexattention import ModelConfig, build_model
    from .expanded_attention import ExpandedConfig, build_expanded_model
    from .cuda_allocation import CudaAllocationTrace
except ImportError:
    from canonical_flexattention import ModelConfig, build_model
    from expanded_attention import ExpandedConfig, build_expanded_model
    from cuda_allocation import CudaAllocationTrace


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
END_TO_END = "end_to_end"
DEVICE_RESIDENT = "device_resident"
_CUDA_REGISTERED = False


def score_matrix_bytes(
    batch: int, query_heads: int, query_length: int, key_length: int
) -> int:
    return batch * query_heads * query_length * key_length * 4


@dataclass(frozen=True)
class DTypeSpec:
    name: str
    tensor_type: int
    numpy_dtype: np.dtype[Any]
    atol: float
    rtol: float


@dataclass
class PreparedBinding:
    binding: Any
    output_ortvalue: Any
    owners: list[Any]
    device_resident: bool
    output_torch: Any | None = None

    def synchronize_inputs(self) -> None:
        if self.device_resident:
            import torch

            torch.cuda.synchronize()
        else:
            self.binding.synchronize_inputs()

    def synchronize_outputs(self) -> None:
        if self.device_resident:
            import torch

            torch.cuda.synchronize()
        else:
            self.binding.synchronize_outputs()

    def output_numpy(self, spec: DTypeSpec) -> np.ndarray:
        if self.output_torch is not None:
            return self.output_torch.float().cpu().numpy().copy()
        if spec.tensor_type == TensorProto.BFLOAT16:
            import torch

            return torch.from_dlpack(self.output_ortvalue).float().numpy().copy()
        return self.output_ortvalue.numpy().copy()


def dtype_spec(name: str) -> DTypeSpec:
    if name == "float32":
        return DTypeSpec(name, TensorProto.FLOAT, np.dtype(np.float32), 1e-5, 1e-4)
    if name == "float16":
        return DTypeSpec(name, TensorProto.FLOAT16, np.dtype(np.float16), 5e-3, 5e-3)
    if name == "bfloat16":
        import ml_dtypes

        return DTypeSpec(
            name,
            TensorProto.BFLOAT16,
            np.dtype(ml_dtypes.bfloat16),
            1e-2,
            1e-2,
        )
    raise ValueError(f"unsupported dtype: {name}")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def register_cuda(ep_library: Path | None) -> None:
    # ORT registers a provider library process-wide, so the whole benchmark run
    # shares a single --ep-library; later calls are intentionally no-ops.
    global _CUDA_REGISTERED
    if _CUDA_REGISTERED:
        return
    if ep_library is None:
        raise ValueError("--ep-library is required for CUDA benchmarks")
    import onnxruntime as ort

    ort.register_execution_provider_library(
        CUDA_PROVIDER, str(ep_library.resolve())
    )
    _CUDA_REGISTERED = True


def make_feeds(
    seq_len: int,
    batch: int,
    query_heads: int,
    kv_heads: int,
    qk_head_size: int,
    value_head_size: int,
    prefix_len: int,
    spec: DTypeSpec,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + seq_len)
    return {
        "Q": rng.normal(
            size=(batch, query_heads, seq_len, qk_head_size)
        ).astype(spec.numpy_dtype),
        "K": rng.normal(
            size=(batch, kv_heads, seq_len, qk_head_size)
        ).astype(spec.numpy_dtype),
        "V": rng.normal(
            size=(batch, kv_heads, seq_len, value_head_size)
        ).astype(spec.numpy_dtype),
        "prefix_len": np.full(batch, prefix_len, dtype=np.int64),
        "q_start": np.zeros(batch, dtype=np.int64),
        "kv_start": np.zeros(batch, dtype=np.int64),
    }


def expanded_label(provider: str, spec: DTypeSpec) -> str:
    if provider == CPU_PROVIDER and spec.tensor_type == TensorProto.BFLOAT16:
        return "expanded_fp32_cast"
    return "expanded"


def build_benchmark_model(
    implementation: str,
    provider: str,
    spec: DTypeSpec,
    seq_len: int,
    batch: int,
    query_heads: int,
    kv_heads: int,
    qk_head_size: int,
    value_head_size: int,
) -> Any:
    if implementation == "direct":
        return build_model(ModelConfig(q_dtype=spec.tensor_type))
    return build_expanded_model(
        ExpandedConfig(
            batch,
            query_heads,
            kv_heads,
            seq_len,
            seq_len,
            qk_head_size,
            value_head_size,
            spec.tensor_type,
            cast_inputs_to_float32=(
                provider == CPU_PROVIDER
                and spec.tensor_type == TensorProto.BFLOAT16
            ),
        )
    )


def session_options(
    cpu_workers: int, trace: CudaAllocationTrace | None = None
) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = cpu_workers
    if trace is not None:
        options.add_session_config_entry(
            "gpu_external_alloc", str(trace.alloc_address)
        )
        options.add_session_config_entry(
            "gpu_external_free", str(trace.free_address)
        )
    return options


def _cpu_ortvalue(value: np.ndarray, spec: DTypeSpec, is_tensor: bool) -> Any:
    import onnxruntime as ort

    if is_tensor and spec.tensor_type == TensorProto.BFLOAT16:
        return ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
            value, TensorProto.BFLOAT16
        )
    return ort.OrtValue.ortvalue_from_numpy(value)


def prepare_cpu_binding(
    session: Any,
    feeds: dict[str, np.ndarray],
    output_shape: tuple[int, ...],
    spec: DTypeSpec,
) -> PreparedBinding:
    binding = session.io_binding()
    owners: list[Any] = []
    for name, value in feeds.items():
        ort_value = _cpu_ortvalue(value, spec, name in {"Q", "K", "V"})
        owners.append(ort_value)
        binding.bind_ortvalue_input(name, ort_value)
    output_array = np.empty(output_shape, dtype=spec.numpy_dtype)
    output_ortvalue = _cpu_ortvalue(output_array, spec, True)
    owners.extend((output_array, output_ortvalue))
    binding.bind_ortvalue_output(
        session.get_outputs()[0].name, output_ortvalue
    )
    return PreparedBinding(binding, output_ortvalue, owners, False)


def _prepare_device_binding(
    session: Any,
    feeds: dict[str, np.ndarray],
    output_shape: tuple[int, ...],
    spec: DTypeSpec,
) -> PreparedBinding:
    import onnxruntime as ort
    import torch

    binding = session.io_binding()
    owners: list[Any] = []
    torch_dtype = (
        torch.bfloat16
        if spec.tensor_type == TensorProto.BFLOAT16
        else torch.float16
    )
    for name, value in feeds.items():
        if name in {"Q", "K", "V"}:
            source = torch.from_numpy(value.astype(np.float32, copy=False))
            tensor = source.to(device="cuda", dtype=torch_dtype)
        else:
            tensor = torch.from_numpy(value).to(device="cuda")
        ort_value = ort.OrtValue.from_dlpack(tensor)
        owners.extend((tensor, ort_value))
        binding.bind_ortvalue_input(name, ort_value)
    output = torch.empty(output_shape, device="cuda", dtype=torch_dtype)
    output_ortvalue = ort.OrtValue.from_dlpack(output)
    owners.extend((output, output_ortvalue))
    binding.bind_ortvalue_output(
        session.get_outputs()[0].name, output_ortvalue
    )
    return PreparedBinding(
        binding,
        output_ortvalue,
        owners,
        True,
        output_torch=output,
    )


def prepare_binding(
    session: Any,
    feeds: dict[str, np.ndarray],
    output_shape: tuple[int, ...],
    spec: DTypeSpec,
    scope: str,
) -> PreparedBinding:
    if scope == DEVICE_RESIDENT:
        return _prepare_device_binding(session, feeds, output_shape, spec)
    return prepare_cpu_binding(session, feeds, output_shape, spec)
