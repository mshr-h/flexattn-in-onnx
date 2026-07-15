from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .benchmark_runtime import (
        CPU_PROVIDER,
        CUDA_PROVIDER,
        build_benchmark_model,
        dtype_spec,
        expanded_label,
        make_feeds,
        prepare_cpu_binding,
        register_cuda,
        score_matrix_bytes,
        session_options,
    )
    from .cuda_allocation import CudaAllocationTrace
except ImportError:
    from benchmark_runtime import (
        CPU_PROVIDER,
        CUDA_PROVIDER,
        build_benchmark_model,
        dtype_spec,
        expanded_label,
        make_feeds,
        prepare_cpu_binding,
        register_cuda,
        score_matrix_bytes,
        session_options,
    )
    from cuda_allocation import CudaAllocationTrace


def _rss_hwm_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/self/status does not contain VmHWM")


def measure_peak_memory(args: argparse.Namespace) -> dict[str, object]:
    import onnxruntime as ort

    spec = dtype_spec(args.dtype)
    ep_library = Path(args.ep_library) if args.ep_library else None
    if args.provider == CUDA_PROVIDER:
        register_cuda(ep_library)
    baseline_rss = _rss_hwm_bytes()
    feeds = make_feeds(
        args.seq_len,
        args.batch,
        args.query_heads,
        args.kv_heads,
        args.qk_head_size,
        args.value_head_size,
        args.prefix_len,
        spec,
        args.seed,
    )
    model = build_benchmark_model(
        args.implementation,
        args.provider,
        spec,
        args.seq_len,
        args.batch,
        args.query_heads,
        args.kv_heads,
        args.qk_head_size,
        args.value_head_size,
    )
    trace = CudaAllocationTrace() if args.provider == CUDA_PROVIDER else None
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=session_options(args.cpu_workers, trace),
        providers=[args.provider],
    )
    output_shape = (
        args.batch,
        args.query_heads,
        args.seq_len,
        args.value_head_size,
    )
    prepared = prepare_cpu_binding(session, feeds, output_shape, spec)
    prepared.synchronize_inputs()
    session.run_with_iobinding(prepared.binding)
    prepared.synchronize_outputs()

    largest_allocation: int | None = None
    if trace is None:
        baseline_memory = baseline_rss
        total_peak_before_run = _rss_hwm_bytes()
    else:
        baseline_memory = trace.active_bytes
        trace.begin_measurement()

    session.run_with_iobinding(prepared.binding)
    prepared.synchronize_outputs()
    if trace is None:
        total_peak = max(total_peak_before_run, _rss_hwm_bytes())
        incremental_peak = max(0, total_peak - baseline_memory)
        metric = "cpu_rss_hwm"
    else:
        total_peak = trace.peak_active_bytes
        incremental_peak = max(
            0, trace.peak_active_bytes - baseline_memory
        )
        largest_allocation = max(trace.allocations, default=0)
        metric = "cuda_external_active"
        if trace.errors:
            raise RuntimeError("; ".join(trace.errors))

    score_bytes = score_matrix_bytes(
        args.batch, args.query_heads, args.seq_len, args.seq_len
    )
    return {
        "provider": args.provider,
        "dtype": args.dtype,
        "seq_len": args.seq_len,
        "implementation": (
            "direct"
            if args.implementation == "direct"
            else expanded_label(args.provider, spec)
        ),
        "memory_metric": metric,
        "peak_memory_baseline_bytes": baseline_memory,
        "peak_memory_total_bytes": total_peak,
        "peak_memory_incremental_bytes": incremental_peak,
        "largest_allocation_bytes": largest_allocation,
        "score_matrix_bytes_float32": score_bytes,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--provider",
        choices=(CPU_PROVIDER, CUDA_PROVIDER),
        required=True,
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        required=True,
    )
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument(
        "--implementation", choices=("direct", "expanded"), required=True
    )
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--query-heads", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--qk-head-size", type=int, required=True)
    parser.add_argument("--value-head-size", type=int, required=True)
    parser.add_argument("--prefix-len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cpu-workers", type=int, required=True)
    parser.add_argument("--ep-library")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.provider == CUDA_PROVIDER and args.dtype == "float32":
        raise ValueError("CUDA dtype matrix supports only float16 and bfloat16")
    result = measure_peak_memory(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
