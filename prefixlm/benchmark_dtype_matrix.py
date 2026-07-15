from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .benchmark_runtime import (
        CPU_PROVIDER,
        CUDA_PROVIDER,
        DEVICE_RESIDENT,
        END_TO_END,
        DTypeSpec,
        PreparedBinding,
        build_benchmark_model as _model,
        dtype_spec,
        expanded_label as _expanded_label,
        make_feeds as _feeds,
        percentile,
        prepare_binding as _prepare_binding,
        prepare_cpu_binding as _prepare_cpu_binding,
        register_cuda as _register_cuda,
        score_matrix_bytes,
        session_options as _session_options,
    )
except ImportError:
    from benchmark_runtime import (
        CPU_PROVIDER,
        CUDA_PROVIDER,
        DEVICE_RESIDENT,
        END_TO_END,
        DTypeSpec,
        PreparedBinding,
        build_benchmark_model as _model,
        dtype_spec,
        expanded_label as _expanded_label,
        make_feeds as _feeds,
        percentile,
        prepare_binding as _prepare_binding,
        prepare_cpu_binding as _prepare_cpu_binding,
        register_cuda as _register_cuda,
        score_matrix_bytes,
        session_options as _session_options,
    )


def _measure(
    session: Any,
    prepared: PreparedBinding,
    warmup: int,
    repeats: int,
) -> list[float]:
    prepared.synchronize_inputs()
    for _ in range(warmup):
        session.run_with_iobinding(prepared.binding)
        prepared.synchronize_outputs()
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        session.run_with_iobinding(prepared.binding)
        prepared.synchronize_outputs()
        durations.append((time.perf_counter() - started) * 1000.0)
    return durations


def _latency_stats(durations: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(durations),
        "p50_ms": statistics.median(durations),
        "p90_ms": percentile(durations, 0.9),
        "min_ms": min(durations),
        "max_ms": max(durations),
    }


def _latency_case(
    args: argparse.Namespace,
    provider: str,
    spec: DTypeSpec,
    seq_len: int,
    scope: str,
) -> list[dict[str, object]]:
    import onnxruntime as ort

    if provider == CUDA_PROVIDER:
        _register_cuda(args.ep_library)
    feeds = _feeds(
        seq_len,
        args.batch,
        args.query_heads,
        args.kv_heads,
        args.qk_head_size,
        args.value_head_size,
        args.prefix_len,
        spec,
        args.seed,
    )
    output_shape = (
        args.batch,
        args.query_heads,
        seq_len,
        args.value_head_size,
    )
    measured: dict[str, tuple[dict[str, float], np.ndarray]] = {}
    implementations = ("direct", _expanded_label(provider, spec))
    for label in implementations:
        model_kind = "direct" if label == "direct" else "expanded"
        model = _model(
            model_kind,
            provider,
            spec,
            seq_len,
            args.batch,
            args.query_heads,
            args.kv_heads,
            args.qk_head_size,
            args.value_head_size,
        )
        session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=_session_options(args.cpu_workers),
            providers=[provider],
        )
        prepared = _prepare_binding(
            session, feeds, output_shape, spec, scope
        )
        durations = _measure(
            session, prepared, args.warmup, args.repeats
        )
        output = prepared.output_numpy(spec)
        measured[label] = (_latency_stats(durations), output)
        del prepared, session
        gc.collect()
        if provider == CUDA_PROVIDER and scope == DEVICE_RESIDENT:
            import torch

            torch.cuda.empty_cache()

    direct_stats, direct_output = measured["direct"]
    expanded_stats, expanded_output = measured[implementations[1]]
    np.testing.assert_allclose(
        direct_output,
        expanded_output,
        atol=spec.atol,
        rtol=spec.rtol,
    )
    max_abs_diff = float(
        np.max(np.abs(direct_output - expanded_output))
    )
    speedup = expanded_stats["p50_ms"] / direct_stats["p50_ms"]
    score_bytes = score_matrix_bytes(
        args.batch, args.query_heads, seq_len, seq_len
    )

    rows = []
    for label in implementations:
        stats, _ = measured[label]
        rows.append(
            {
                "provider": provider,
                "dtype": spec.name,
                "scope": scope,
                "seq_len": seq_len,
                "implementation": label,
                **stats,
                "p50_speedup_expanded_over_direct": speedup,
                "max_abs_diff": max_abs_diff,
                "score_matrix_bytes_float32": score_bytes,
            }
        )
    return rows


def _validate_assignment(
    args: argparse.Namespace, provider: str, spec: DTypeSpec
) -> dict[str, object]:
    import onnxruntime as ort

    if provider == CUDA_PROVIDER:
        _register_cuda(args.ep_library)
    seq_len = max(64, args.prefix_len)
    feeds = _feeds(
        seq_len,
        args.batch,
        args.query_heads,
        args.kv_heads,
        args.qk_head_size,
        args.value_head_size,
        args.prefix_len,
        spec,
        args.seed,
    )
    output_shape = (
        args.batch,
        args.query_heads,
        seq_len,
        args.value_head_size,
    )
    with tempfile.TemporaryDirectory(prefix="prefixlm-profile-") as directory:
        options = _session_options(args.cpu_workers)
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(directory) / "profile")
        model = _model(
            "direct",
            provider,
            spec,
            seq_len,
            args.batch,
            args.query_heads,
            args.kv_heads,
            args.qk_head_size,
            args.value_head_size,
        )
        session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=[provider],
        )
        prepared = _prepare_cpu_binding(
            session, feeds, output_shape, spec
        )
        session.run_with_iobinding(prepared.binding)
        prepared.synchronize_outputs()
        profile_path = Path(session.end_profiling())
        events = json.loads(profile_path.read_text())
    hits = [
        event
        for event in events
        if event.get("cat") == "Node"
        and event.get("args", {}).get("op_name") == "FlexAttention"
        and event.get("name", "").endswith("_kernel_time")
    ]
    providers = {event.get("args", {}).get("provider") for event in hits}
    if len(hits) != 1 or providers != {provider}:
        raise AssertionError(
            f"expected one FlexAttention event on {provider}, "
            f"got count={len(hits)}, providers={sorted(providers)}"
        )
    return {
        "provider": provider,
        "dtype": spec.name,
        "flexattention_events": 1,
        "assigned_provider": provider,
    }


def _invoke_memory_worker(
    args: argparse.Namespace,
    provider: str,
    dtype: str,
    seq_len: int,
    implementation: str,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="prefixlm-memory-") as directory:
        output = Path(directory) / "result.json"
        command = [
            sys.executable,
            str(
                Path(__file__)
                .with_name("benchmark_peak_memory.py")
                .resolve()
            ),
            "--output",
            str(output),
            "--provider",
            provider,
            "--dtype",
            dtype,
            "--seq-len",
            str(seq_len),
            "--implementation",
            implementation,
            "--batch",
            str(args.batch),
            "--query-heads",
            str(args.query_heads),
            "--kv-heads",
            str(args.kv_heads),
            "--qk-head-size",
            str(args.qk_head_size),
            "--value-head-size",
            str(args.value_head_size),
            "--prefix-len",
            str(args.prefix_len),
            "--seed",
            str(args.seed),
            "--cpu-workers",
            str(args.cpu_workers),
        ]
        if args.ep_library is not None:
            command.extend(("--ep-library", str(args.ep_library.resolve())))
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"memory worker failed ({completed.returncode}): "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        return json.loads(output.read_text())


def _environment() -> dict[str, object]:
    import onnxruntime as ort

    environment: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "logical_cpus": os.cpu_count(),
    }
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version",
        "--format=csv,noheader",
    ]
    try:
        environment["gpu"] = subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        environment["gpu"] = None
    return environment


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_summary(payload: dict[str, object]) -> str:
    lines = [
        "# PrefixLM FlexAttention dtype matrix",
        "",
        "Latency values are milliseconds. Speedup is expanded p50/direct p50.",
        "",
    ]
    latency_rows = payload["latency_results"]
    assert isinstance(latency_rows, list)
    grouped: dict[tuple[str, str, str, int], dict[str, dict[str, object]]] = {}
    for row in latency_rows:
        key = (
            str(row["provider"]),
            str(row["dtype"]),
            str(row["scope"]),
            int(row["seq_len"]),
        )
        grouped.setdefault(key, {})[str(row["implementation"])] = row
    for provider, dtype, scope in sorted(
        {(key[0], key[1], key[2]) for key in grouped}
    ):
        lines.extend(
            [
                f"## {provider} / {dtype} / {scope}",
                "",
                "| S | direct p50 | p90 | mean | expanded p50 | p90 | mean | "
                "speedup | max abs diff |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key in sorted(
            (key for key in grouped if key[:3] == (provider, dtype, scope)),
            key=lambda item: item[3],
        ):
            rows = grouped[key]
            direct = rows.get("direct")
            expanded_name = next(
                (name for name in rows if name != "direct"), None
            )
            if direct is None or expanded_name is None:
                continue
            expanded = rows[expanded_name]
            lines.append(
                f"| {key[3]} | {direct['p50_ms']:.6f} | "
                f"{direct['p90_ms']:.6f} | {direct['mean_ms']:.6f} | "
                f"{expanded['p50_ms']:.6f} | {expanded['p90_ms']:.6f} | "
                f"{expanded['mean_ms']:.6f} | "
                f"{direct['p50_speedup_expanded_over_direct']:.3f} | "
                f"{direct['max_abs_diff']:.6g} |"
            )
        lines.append("")

    memory_rows = payload["memory_results"]
    assert isinstance(memory_rows, list)
    memory_grouped: dict[
        tuple[str, str, int], dict[str, dict[str, object]]
    ] = {}
    for row in memory_rows:
        key = (
            str(row["provider"]),
            str(row["dtype"]),
            int(row["seq_len"]),
        )
        memory_grouped.setdefault(key, {})[str(row["implementation"])] = row
    for provider, dtype in sorted(
        {(key[0], key[1]) for key in memory_grouped}
    ):
        lines.extend(
            [
                f"## Memory / {provider} / {dtype}",
                "",
                "| S | score | direct total | "
                "direct delta | direct largest | direct gate | expanded total | "
                "expanded delta | expanded largest | expanded gate |",
                "|---:|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|",
            ]
        )
        for key in sorted(
            (
                key
                for key in memory_grouped
                if key[:2] == (provider, dtype)
            ),
            key=lambda item: item[2],
        ):
            rows = memory_grouped[key]
            direct = rows.get("direct")
            expanded_name = next(
                (name for name in rows if name != "direct"), None
            )
            if direct is None or expanded_name is None:
                continue
            expanded = rows[expanded_name]
            direct_largest = direct["largest_allocation_bytes"]
            expanded_largest = expanded["largest_allocation_bytes"]
            direct_largest_text = (
                "n/a"
                if direct_largest is None
                else f"{direct_largest / 2**20:.3f} MiB"
            )
            expanded_largest_text = (
                "n/a"
                if expanded_largest is None
                else f"{expanded_largest / 2**20:.3f} MiB"
            )
            direct_gate = direct["score_buffer_gate_passed"]
            expanded_gate = expanded["score_buffer_gate_passed"]
            direct_gate_text = (
                "n/a" if direct_gate is None else "pass" if direct_gate else "fail"
            )
            expanded_gate_text = (
                "n/a"
                if expanded_gate is None
                else "pass" if expanded_gate else "fail"
            )
            lines.append(
                f"| {key[2]} | "
                f"{expanded['score_matrix_bytes_float32'] / 2**20:.3f} MiB | "
                f"{direct['peak_memory_total_bytes'] / 2**20:.3f} MiB | "
                f"{direct['peak_memory_incremental_bytes'] / 2**20:.3f} MiB | "
                f"{direct_largest_text} | "
                f"{direct_gate_text} | "
                f"{expanded['peak_memory_total_bytes'] / 2**20:.3f} MiB | "
                f"{expanded['peak_memory_incremental_bytes'] / 2**20:.3f} MiB | "
                f"{expanded_largest_text} | {expanded_gate_text} |"
            )
        lines.append("")

    failures = payload["failures"]
    assert isinstance(failures, list)
    lines.extend(["## Failures", ""])
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _benchmark_matrix(
    providers: list[str],
) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]:
    matrix = []
    if "cpu" in providers:
        matrix.append(
            (CPU_PROVIDER, ("float32", "float16", "bfloat16"), (END_TO_END,))
        )
    if "cuda" in providers:
        matrix.append(
            (
                CUDA_PROVIDER,
                ("float16", "bfloat16"),
                (END_TO_END, DEVICE_RESIDENT),
            )
        )
    return tuple(matrix)


def _append_completeness_failures(
    args: argparse.Namespace,
    assignments: list[dict[str, object]],
    latency_results: list[dict[str, object]],
    memory_results: list[dict[str, object]],
    failures: list[str],
) -> None:
    for row in memory_results:
        row["score_buffer_gate_passed"] = None
        row["score_buffer_gate_rule"] = "not_applicable"

    matrix = _benchmark_matrix(args.providers)
    expected_assignments = sum(len(dtypes) for _, dtypes, _ in matrix)
    expected_latency = len(args.seq_lens) * sum(
        len(dtypes) * len(scopes) * 2
        for _, dtypes, scopes in matrix
    )
    expected_memory = len(args.seq_lens) * sum(
        len(dtypes) * 2 for _, dtypes, _ in matrix
    )
    if len(assignments) != expected_assignments:
        failures.append(
            f"expected {expected_assignments} assignment rows, "
            f"got {len(assignments)}"
        )
    if len(latency_results) != expected_latency:
        failures.append(
            f"expected {expected_latency} latency rows, "
            f"got {len(latency_results)}"
        )
    if len(memory_results) != expected_memory:
        failures.append(
            f"expected {expected_memory} memory rows, "
            f"got {len(memory_results)}"
        )

    by_key = {
        (
            str(row["provider"]),
            str(row["dtype"]),
            int(row["seq_len"]),
            str(row["implementation"]),
        ): row
        for row in memory_results
    }
    if "cuda" in args.providers:
        for dtype in ("float16", "bfloat16"):
            for seq_len in args.seq_lens:
                score_bytes = score_matrix_bytes(
                    args.batch, args.query_heads, seq_len, seq_len
                )
                direct = by_key.get(
                    (CUDA_PROVIDER, dtype, seq_len, "direct")
                )
                expanded = by_key.get(
                    (CUDA_PROVIDER, dtype, seq_len, "expanded")
                )
                if direct is not None:
                    largest = int(direct["largest_allocation_bytes"] or 0)
                    passed = largest < score_bytes
                    direct["score_buffer_gate_passed"] = passed
                    direct["score_buffer_gate_rule"] = (
                        "largest_allocation_below_float32_score_buffer"
                    )
                    if not passed:
                        failures.append(
                            f"CUDA {dtype} S={seq_len} direct allocation "
                            f"{largest} reached score size {score_bytes}"
                        )
                if expanded is not None:
                    largest = int(expanded["largest_allocation_bytes"] or 0)
                    passed = largest >= score_bytes
                    expanded["score_buffer_gate_passed"] = passed
                    expanded["score_buffer_gate_rule"] = (
                        "largest_allocation_reaches_float32_score_buffer"
                    )
                    if not passed:
                        failures.append(
                            f"CUDA {dtype} S={seq_len} expanded allocation "
                            f"{largest} was below score size {score_bytes}"
                        )

    if "cpu" not in args.providers:
        return
    for seq_len in args.seq_lens:
        if seq_len < 2048:
            continue
        score_bytes = score_matrix_bytes(
            args.batch, args.query_heads, seq_len, seq_len
        )
        for dtype in ("float32", "float16", "bfloat16"):
            direct = by_key.get(
                (CPU_PROVIDER, dtype, seq_len, "direct")
            )
            expanded_name = (
                "expanded_fp32_cast" if dtype == "bfloat16" else "expanded"
            )
            expanded = by_key.get(
                (CPU_PROVIDER, dtype, seq_len, expanded_name)
            )
            if direct is not None:
                direct_delta = int(direct["peak_memory_incremental_bytes"])
                passed = direct_delta < score_bytes
                direct["score_buffer_gate_passed"] = passed
                direct["score_buffer_gate_rule"] = (
                    "incremental_peak_below_float32_score_buffer"
                )
                if not passed:
                    failures.append(
                        f"CPU {dtype} S={seq_len} direct peak delta "
                        f"{direct_delta} reached score size {score_bytes}"
                    )
            if expanded is not None:
                expanded_delta = int(expanded["peak_memory_incremental_bytes"])
                passed = expanded_delta >= score_bytes
                expanded["score_buffer_gate_passed"] = passed
                expanded["score_buffer_gate_rule"] = (
                    "incremental_peak_reaches_float32_score_buffer"
                )
                if not passed:
                    failures.append(
                        f"CPU {dtype} S={seq_len} expanded peak delta "
                        f"{expanded_delta} was below score size {score_bytes}"
                    )


def _matrix_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[128, 512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--qk-head-size", type=int, default=64)
    parser.add_argument("--value-head-size", type=int, default=64)
    parser.add_argument("--prefix-len", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=("cpu", "cuda"),
        default=["cpu", "cuda"],
    )
    parser.add_argument("--ep-library", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("prefixlm/benchmark_results/dtype_matrix"),
    )
    return parser


def run_matrix(args: argparse.Namespace) -> dict[str, object]:
    if any(seq_len < args.prefix_len for seq_len in args.seq_lens):
        raise ValueError("every seq-len must be at least prefix-len")
    if args.query_heads % args.kv_heads:
        raise ValueError("query-heads must be divisible by kv-heads")
    if "cuda" in args.providers and args.ep_library is None:
        raise ValueError("--ep-library is required when CUDA is selected")
    matrix = _benchmark_matrix(args.providers)
    latency_results: list[dict[str, object]] = []
    memory_results: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    failures: list[str] = []
    for provider, dtypes, scopes in matrix:
        for dtype in dtypes:
            spec = dtype_spec(dtype)
            try:
                assignments.append(
                    _validate_assignment(args, provider, spec)
                )
            except Exception as error:
                failures.append(
                    f"assignment {provider}/{dtype}: {error}"
                )
            for seq_len in args.seq_lens:
                for scope in scopes:
                    try:
                        latency_results.extend(
                            _latency_case(
                                args, provider, spec, seq_len, scope
                            )
                        )
                    except Exception as error:
                        failures.append(
                            f"latency {provider}/{dtype}/S={seq_len}/"
                            f"{scope}: {error}"
                        )
                for implementation in ("direct", "expanded"):
                    try:
                        memory_results.append(
                            _invoke_memory_worker(
                                args,
                                provider,
                                dtype,
                                seq_len,
                                implementation,
                            )
                        )
                    except Exception as error:
                        failures.append(
                            f"memory {provider}/{dtype}/S={seq_len}/"
                            f"{implementation}: {error}"
                        )
    _append_completeness_failures(
        args,
        assignments,
        latency_results,
        memory_results,
        failures,
    )
    return {
        "environment": _environment(),
        "command": " ".join(shlex.quote(argument) for argument in sys.argv),
        "config": {
            "seq_lens": args.seq_lens,
            "batch": args.batch,
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "qk_head_size": args.qk_head_size,
            "value_head_size": args.value_head_size,
            "prefix_len": args.prefix_len,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "cpu_workers": args.cpu_workers,
            "providers": args.providers,
        },
        "assignments": assignments,
        "latency_results": latency_results,
        "memory_results": memory_results,
        "failures": failures,
    }


def main() -> None:
    args = _matrix_parser().parse_args()
    payload = run_matrix(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    _write_csv(args.out_dir / "latency.csv", payload["latency_results"])
    _write_csv(args.out_dir / "memory.csv", payload["memory_results"])
    (args.out_dir / "summary.md").write_text(
        _markdown_summary(payload)
    )
    print(json.dumps(payload, indent=2))
    if payload["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
