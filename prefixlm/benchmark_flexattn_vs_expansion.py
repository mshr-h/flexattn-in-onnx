from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import checker

from compare_flexattn_pytorch_ort import (
    count_flex_attention_nodes,
    default_tolerances,
    numpy_dtype,
)
from export_flexattn_onnx import (
    MAIN_OPSET,
    PREVIEW_DOMAIN,
    annotate_score_mod_types,
    export_model_proto,
    verify_model,
)
from model import build_model


GRAPH_OPTIMIZATION_LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


class MissingPreviewFlexAttentionRuntime(RuntimeError):
    pass


@dataclass
class BenchmarkRow:
    implementation: str
    status: str
    batch: int
    seq_len: int
    prefix_len: int
    embed_dim: int
    num_heads: int
    head_dim: int
    dtype: str
    provider: str
    graph_optimization_level: str
    intra_op_num_threads: int | str
    inter_op_num_threads: int | str
    warmup: int
    repeats: int
    mean_ms: float | str = ""
    p50_ms: float | str = ""
    p90_ms: float | str = ""
    min_ms: float | str = ""
    max_ms: float | str = ""
    std_ms: float | str = ""
    speedup_vs_expanded_or_empty: float | str = ""
    max_abs_diff: float | str = ""
    mean_abs_diff: float | str = ""
    allclose: bool | str = ""
    rtol: float | str = ""
    atol: float | str = ""
    error_or_empty: str = ""
    optimized_op_counts_or_empty: str = ""
    profile_top_ops_or_empty: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PrefixLM FlexAttention ONNX Runtime CPU EP against an "
            "expanded lower-level ONNX graph."
        )
    )
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 1024, 2048])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=3)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=Path("prefixlm/benchmark_results"))
    parser.add_argument("--max-score-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--enable-profiling", action="store_true")
    parser.add_argument("--save-optimized-models", action="store_true")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--input-seed", type=int, default=0)
    parser.add_argument("--opset", type=int, default=MAIN_OPSET)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--intra-op-num-threads", type=int, default=0)
    parser.add_argument("--inter-op-num-threads", type=int, default=0)
    parser.add_argument(
        "--graph-optimization-level",
        choices=tuple(GRAPH_OPTIMIZATION_LEVELS),
        default="all",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if not args.seq_lens or any(seq_len <= 0 for seq_len in args.seq_lens):
        raise ValueError("--seq-lens must contain positive values")
    if args.prefix_len < 0:
        raise ValueError("--prefix-len must be non-negative")
    if any(args.prefix_len > seq_len for seq_len in args.seq_lens):
        raise ValueError("--prefix-len must be <= every sequence length")
    if args.embed_dim <= 0:
        raise ValueError("--embed-dim must be positive")
    if args.num_heads <= 0:
        raise ValueError("--num-heads must be positive")
    if args.embed_dim % args.num_heads != 0:
        raise ValueError("--embed-dim must be divisible by --num-heads")
    if args.embed_dim // args.num_heads % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.max_score_bytes < 0:
        raise ValueError("--max-score-bytes must be non-negative")
    if args.opset < 18:
        raise ValueError("--opset must be at least 18 for the dynamo exporter path")


def diagnostics_dir(args: argparse.Namespace) -> Path:
    return args.out_dir / "diagnostics"


def make_session_options(
    args: argparse.Namespace, implementation: str, seq_len: int
) -> tuple[ort.SessionOptions, Path | None]:
    options = ort.SessionOptions()
    options.graph_optimization_level = GRAPH_OPTIMIZATION_LEVELS[
        args.graph_optimization_level
    ]
    if args.intra_op_num_threads:
        options.intra_op_num_threads = args.intra_op_num_threads
    if args.inter_op_num_threads:
        options.inter_op_num_threads = args.inter_op_num_threads

    optimized_model_path = None
    if args.enable_profiling or args.save_optimized_models:
        diagnostics_dir(args).mkdir(parents=True, exist_ok=True)
    if args.enable_profiling:
        options.enable_profiling = True
        options.profile_file_prefix = str(
            diagnostics_dir(args) / f"{implementation}_seq{seq_len}_profile"
        )
    if args.save_optimized_models:
        optimized_model_path = diagnostics_dir(args) / (
            f"{implementation}_seq{seq_len}_optimized.onnx"
        )
        options.optimized_model_filepath = str(optimized_model_path)
    return options, optimized_model_path


def is_missing_preview_flexattention_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "flexattention" in message and any(
        hint in message
        for hint in (
            "no op registered",
            "not a registered",
            "not registered",
            "kernel not found",
            "invalidgraph",
            "fatal error",
        )
    )


def missing_preview_flexattention_message(exc: Exception) -> str:
    return (
        f"ONNX Runtime CPUExecutionProvider could not load or run "
        f"{PREVIEW_DOMAIN}::FlexAttention. A preview ONNX Runtime build with "
        f"FlexAttention support is required. Loaded onnxruntime from: {ort.__file__}. "
        f"Original error: {exc}"
    )


def export_expanded_model_proto(
    model: torch.nn.Module,
    dummy_x: torch.Tensor,
    opset: int,
) -> onnx.ModelProto:
    with torch.no_grad():
        onnx_program = torch.onnx.export(
            model,
            (dummy_x,),
            f=None,
            input_names=["x"],
            output_names=["y"],
            opset_version=opset,
            dynamo=True,
            optimize=False,
        )
    if onnx_program is None:
        raise RuntimeError(
            "torch.onnx.export(dynamo=True) did not return an ONNXProgram"
        )
    model_proto = onnx_program.model_proto
    checker.check_model(model_proto)
    flex_count = count_flex_attention_nodes(model_proto)
    if flex_count != 0:
        raise ValueError(f"Expected zero FlexAttention nodes, found {flex_count}")
    ops = {node.op_type for node in model_proto.graph.node}
    missing_ops = {"MatMul", "Softmax"} - ops
    if missing_ops:
        raise ValueError(f"Expanded model is missing expected ops: {sorted(missing_ops)}")
    return model_proto


def export_flex_checked(
    model: torch.nn.Module,
    dummy_x: torch.Tensor,
    opset: int,
) -> onnx.ModelProto:
    model_proto = export_model_proto(model, dummy_x, opset)
    annotate_score_mod_types(model_proto)
    verify_model(model_proto)
    flex_count = count_flex_attention_nodes(model_proto)
    if flex_count != 1:
        raise ValueError(f"Expected one FlexAttention node, found {flex_count}")
    return model_proto


def format_mapping(mapping: dict[str, float | int], limit: int = 8) -> str:
    items = sorted(mapping.items(), key=lambda item: item[1], reverse=True)[:limit]
    formatted = []
    for name, value in items:
        if isinstance(value, float):
            formatted.append(f"{name}={value:.3f}")
        else:
            formatted.append(f"{name}={value}")
    return "; ".join(formatted)


def optimized_op_counts(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    model = onnx.load(path)
    counts = Counter(
        f"{node.domain or 'ai.onnx'}::{node.op_type}" for node in model.graph.node
    )
    return format_mapping(dict(counts), limit=10)


def profile_top_ops(profile_path: str) -> str:
    if not profile_path:
        return ""
    events = json.loads(Path(profile_path).read_text())
    durations_ms: dict[str, float] = defaultdict(float)
    for event in events:
        if event.get("cat") != "Node":
            continue
        args = event.get("args", {})
        op_name = args.get("op_name")
        if not op_name:
            op_name = event.get("name", "").split("_kernel_time")[0]
        durations_ms[op_name] += float(event.get("dur", 0)) / 1000.0
    return format_mapping(durations_ms, limit=10)


def make_session(
    model_proto: onnx.ModelProto,
    args: argparse.Namespace,
    implementation: str,
    seq_len: int,
) -> tuple[ort.InferenceSession, Path | None]:
    session_options, optimized_model_path = make_session_options(
        args, implementation, seq_len
    )
    return (
        ort.InferenceSession(
            model_proto.SerializeToString(),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        ),
        optimized_model_path,
    )


def finish_session_diagnostics(
    session: ort.InferenceSession,
    optimized_model_path: Path | None,
    args: argparse.Namespace,
) -> tuple[str, str]:
    profile = ""
    if args.enable_profiling:
        profile = profile_top_ops(session.end_profiling())
    return optimized_op_counts(optimized_model_path), profile


def run_timed(
    session: ort.InferenceSession,
    x_np: np.ndarray,
    warmup: int,
    repeats: int,
) -> tuple[np.ndarray, list[float]]:
    input_name = session.get_inputs()[0].name
    feeds = {input_name: x_np}
    output = None
    for _ in range(warmup):
        output = session.run(None, feeds)[0]

    latencies_ms = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        output = session.run(None, feeds)[0]
        end = time.perf_counter_ns()
        latencies_ms.append((end - start) / 1_000_000.0)

    if output is None:
        output = session.run(None, feeds)[0]
    return output, latencies_ms


def latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    sorted_values = sorted(latencies_ms)
    p90_index = min(len(sorted_values) - 1, int(np.ceil(0.9 * len(sorted_values))) - 1)
    return {
        "mean_ms": float(statistics.fmean(latencies_ms)),
        "p50_ms": float(statistics.median(latencies_ms)),
        "p90_ms": float(sorted_values[p90_index]),
        "min_ms": float(min(latencies_ms)),
        "max_ms": float(max(latencies_ms)),
        "std_ms": float(statistics.pstdev(latencies_ms)),
    }


def compare_outputs(
    flex_output: np.ndarray,
    expanded_output: np.ndarray,
    rtol: float,
    atol: float,
) -> dict[str, float | bool]:
    flex_f32 = flex_output.astype(np.float32)
    expanded_f32 = expanded_output.astype(np.float32)
    diff = np.abs(flex_f32 - expanded_f32)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose": bool(np.allclose(flex_f32, expanded_f32, rtol=rtol, atol=atol)),
    }


def score_bytes(batch: int, num_heads: int, seq_len: int, dtype: str) -> int:
    return batch * num_heads * seq_len * seq_len * numpy_dtype(dtype).itemsize


def base_row(args: argparse.Namespace, seq_len: int, implementation: str) -> BenchmarkRow:
    threads_intra: int | str = args.intra_op_num_threads or "default"
    threads_inter: int | str = args.inter_op_num_threads or "default"
    return BenchmarkRow(
        implementation=implementation,
        status="ok",
        batch=args.batch,
        seq_len=seq_len,
        prefix_len=args.prefix_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        head_dim=args.embed_dim // args.num_heads,
        dtype=args.dtype,
        provider="CPUExecutionProvider",
        graph_optimization_level=args.graph_optimization_level,
        intra_op_num_threads=threads_intra,
        inter_op_num_threads=threads_inter,
        warmup=args.warmup,
        repeats=args.repeats,
    )


def build_pair(
    args: argparse.Namespace, seq_len: int
) -> tuple[torch.nn.Module, torch.nn.Module]:
    flex_model = build_model(
        batch=args.batch,
        seq_len=seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        prefix_len=args.prefix_len,
        dtype=args.dtype,
        seed=args.model_seed,
        implementation="flex",
    )
    expanded_model = build_model(
        batch=args.batch,
        seq_len=seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        prefix_len=args.prefix_len,
        dtype=args.dtype,
        seed=args.model_seed,
        implementation="expanded",
    )
    expanded_model.load_state_dict(flex_model.state_dict(), strict=True)
    return flex_model, expanded_model


def save_model_if_requested(
    args: argparse.Namespace,
    model_proto: onnx.ModelProto,
    implementation: str,
    seq_len: int,
) -> None:
    if not args.save_models:
        return
    model_dir = args.out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(model_proto, model_dir / f"{implementation}_seq{seq_len}.onnx")


def benchmark_seq_len(args: argparse.Namespace, seq_len: int) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    rtol_default, atol_default = default_tolerances(args.dtype)
    rtol = rtol_default if args.rtol is None else args.rtol
    atol = atol_default if args.atol is None else args.atol

    flex_row = base_row(args, seq_len, "flex")
    expanded_row = base_row(args, seq_len, "expanded")

    try:
        flex_model, expanded_model = build_pair(args, seq_len)
        input_shape = (args.batch, seq_len, args.embed_dim)
        x_np = (
            np.random.default_rng(args.input_seed + seq_len)
            .standard_normal(input_shape)
            .astype(numpy_dtype(args.dtype))
        )
        x_torch = torch.from_numpy(x_np.copy())

        flex_proto = export_flex_checked(flex_model, x_torch, args.opset)
        save_model_if_requested(args, flex_proto, "flex", seq_len)
        flex_session, flex_optimized_path = make_session(
            flex_proto, args, "flex", seq_len
        )
        flex_output, flex_latencies = run_timed(
            flex_session, x_np, args.warmup, args.repeats
        )
        for name, value in latency_stats(flex_latencies).items():
            setattr(flex_row, name, value)
        (
            flex_row.optimized_op_counts_or_empty,
            flex_row.profile_top_ops_or_empty,
        ) = finish_session_diagnostics(flex_session, flex_optimized_path, args)
    except Exception as exc:
        if is_missing_preview_flexattention_error(exc):
            raise MissingPreviewFlexAttentionRuntime(
                missing_preview_flexattention_message(exc)
            ) from exc
        flex_row.status = "error"
        flex_row.error_or_empty = str(exc)
        rows.append(flex_row)
        expanded_row.status = "error"
        expanded_row.error_or_empty = "flex variant failed; expanded comparison not run"
        rows.append(expanded_row)
        return rows

    estimated_score_bytes = score_bytes(args.batch, args.num_heads, seq_len, args.dtype)
    if estimated_score_bytes > args.max_score_bytes:
        expanded_row.status = "skipped"
        expanded_row.error_or_empty = (
            f"expanded dense score tensor would require {estimated_score_bytes} bytes, "
            f"exceeding --max-score-bytes={args.max_score_bytes}"
        )
        rows.extend([flex_row, expanded_row])
        return rows

    try:
        expanded_proto = export_expanded_model_proto(expanded_model, x_torch, args.opset)
        save_model_if_requested(args, expanded_proto, "expanded", seq_len)
        expanded_session, expanded_optimized_path = make_session(
            expanded_proto, args, "expanded", seq_len
        )
        expanded_output, expanded_latencies = run_timed(
            expanded_session, x_np, args.warmup, args.repeats
        )
        for name, value in latency_stats(expanded_latencies).items():
            setattr(expanded_row, name, value)
        (
            expanded_row.optimized_op_counts_or_empty,
            expanded_row.profile_top_ops_or_empty,
        ) = finish_session_diagnostics(
            expanded_session, expanded_optimized_path, args
        )

        comparison = compare_outputs(flex_output, expanded_output, rtol=rtol, atol=atol)
        speedup = float(expanded_row.mean_ms) / float(flex_row.mean_ms)
        for row in (flex_row, expanded_row):
            row.max_abs_diff = comparison["max_abs_diff"]
            row.mean_abs_diff = comparison["mean_abs_diff"]
            row.allclose = comparison["allclose"]
            row.rtol = rtol
            row.atol = atol
        flex_row.speedup_vs_expanded_or_empty = speedup
        if not comparison["allclose"]:
            error = (
                "flex and expanded ORT outputs are not allclose "
                f"with rtol={rtol} atol={atol}"
            )
            for row in (flex_row, expanded_row):
                row.status = "error"
                row.error_or_empty = error
    except Exception as exc:
        expanded_row.status = "error"
        expanded_row.error_or_empty = str(exc)

    rows.extend([flex_row, expanded_row])
    return rows


def write_csv(rows: list[BenchmarkRow], path: Path) -> None:
    fieldnames = (
        list(asdict(rows[0]).keys()) if rows else list(BenchmarkRow.__annotations__)
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(rows: list[BenchmarkRow], path: Path, metadata: dict[str, Any]) -> None:
    payload = {"metadata": metadata, "rows": [asdict(row) for row in rows]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(rows: list[BenchmarkRow], path: Path, metadata: dict[str, Any]) -> None:
    lines = [
        "# PrefixLM FlexAttention CPU Benchmark",
        "",
        "## Environment",
        "",
        f"- command: `{' '.join(sys.argv)}`",
        f"- python: `{platform.python_version()}`",
        f"- platform: `{platform.platform()}`",
        f"- onnxruntime_version: `{metadata['onnxruntime_version']}`",
        f"- onnxruntime_file: `{metadata['onnxruntime_file']}`",
        f"- providers: `{metadata['providers']}`",
        "",
        "## Configuration",
        "",
        "| batch | seq_lens | prefix_len | embed_dim | num_heads | dtype | warmup | repeats | graph_optimization_level | intra_op_num_threads | inter_op_num_threads |",
        "|---:|---|---:|---:|---:|---|---:|---:|---|---|---|",
        (
            f"| {metadata['batch']} | {metadata['seq_lens']} | {metadata['prefix_len']} | "
            f"{metadata['embed_dim']} | {metadata['num_heads']} | {metadata['dtype']} | "
            f"{metadata['warmup']} | {metadata['repeats']} | {metadata['graph_optimization_level']} | "
            f"{metadata['intra_op_num_threads']} | {metadata['inter_op_num_threads']} |"
        ),
        "",
        "## Results",
        "",
        "| seq_len | implementation | status | mean_ms | p50_ms | p90_ms | speedup | max_abs_diff | allclose | error |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.seq_len),
                    row.implementation,
                    row.status,
                    format_value(row.mean_ms),
                    format_value(row.p50_ms),
                    format_value(row.p90_ms),
                    format_value(row.speedup_vs_expanded_or_empty),
                    format_value(row.max_abs_diff),
                    format_value(row.allclose),
                    str(row.error_or_empty).replace("\n", " "),
                ]
            )
            + " |"
        )
    if any(
        row.optimized_op_counts_or_empty or row.profile_top_ops_or_empty
        for row in rows
    ):
        lines.extend(
            [
                "",
                "## Diagnostics",
                "",
                "| seq_len | implementation | optimized op counts | profile top ops ms |",
                "|---:|---|---|---|",
            ]
        )
        for row in rows:
            if row.optimized_op_counts_or_empty or row.profile_top_ops_or_empty:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.seq_len),
                            row.implementation,
                            str(row.optimized_op_counts_or_empty),
                            str(row.profile_top_ops_or_empty),
                        ]
                    )
                    + " |"
                )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `speedup` is `expanded_mean_ms / flex_mean_ms`; values above 1 mean FlexAttention was faster.",
            "- Skipped expanded rows exceeded the configured dense score tensor memory guard.",
            "- Prefer p50/p90 over mean when outliers make mean and median disagree.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def print_table(rows: list[BenchmarkRow]) -> None:
    headers = ["seq", "impl", "status", "mean", "p50", "p90", "speedup", "allclose"]
    print(" ".join(f"{header:>12}" for header in headers))
    for row in rows:
        values = [
            row.seq_len,
            row.implementation,
            row.status,
            format_value(row.mean_ms),
            format_value(row.p50_ms),
            format_value(row.p90_ms),
            format_value(row.speedup_vs_expanded_or_empty),
            format_value(row.allclose),
        ]
        print(" ".join(f"{str(value):>12}" for value in values))


def metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "command": sys.argv,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "onnxruntime_version": ort.__version__,
        "onnxruntime_file": ort.__file__,
        "providers": ort.get_available_providers(),
        "batch": args.batch,
        "seq_lens": args.seq_lens,
        "prefix_len": args.prefix_len,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "graph_optimization_level": args.graph_optimization_level,
        "intra_op_num_threads": args.intra_op_num_threads or "default",
        "inter_op_num_threads": args.inter_op_num_threads or "default",
        "max_score_bytes": args.max_score_bytes,
        "opset": args.opset,
        "enable_profiling": args.enable_profiling,
        "save_optimized_models": args.save_optimized_models,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[BenchmarkRow] = []
    try:
        for seq_len in args.seq_lens:
            print(f"benchmarking seq_len={seq_len}", flush=True)
            rows.extend(benchmark_seq_len(args, seq_len))
    except MissingPreviewFlexAttentionRuntime as exc:
        raise SystemExit(str(exc)) from exc

    meta = metadata(args)
    write_csv(rows, args.out_dir / "benchmark_results.csv")
    write_json(rows, args.out_dir / "benchmark_results.json", meta)
    write_markdown(rows, args.out_dir / "benchmark_summary.md", meta)
    print_table(rows)
    print(f"wrote results to {args.out_dir}")
    if any(row.status == "error" for row in rows):
        print(
            "one or more benchmark rows failed; see error_or_empty in the results",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
