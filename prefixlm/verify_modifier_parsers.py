from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import onnx

try:
    from .modifier_corpus import CASES, models
    from .preflight_flexattention import PreflightError, validate_model
except ImportError:
    from modifier_corpus import CASES, models
    from preflight_flexattention import PreflightError, validate_model


def verify(provider: str, ep_library: Path | None) -> None:
    import onnxruntime as ort

    if ep_library is not None:
        ort.register_execution_provider_library(provider, str(ep_library.resolve()))

    for case, model in models():
        preflight_accepted = True
        try:
            validate_model(model)
        except PreflightError:
            preflight_accepted = False
        if preflight_accepted != case.accepted:
            raise AssertionError(
                f"Python preflight disagreed with corpus case {case.name!r}"
            )

        adapter_accepted = False
        with tempfile.TemporaryDirectory(prefix="modifier-corpus-") as directory:
            optimized_path = Path(directory) / "optimized.onnx"
            options = ort.SessionOptions()
            options.optimized_model_filepath = str(optimized_path)
            try:
                ort.InferenceSession(
                    model.SerializeToString(),
                    sess_options=options,
                    providers=[provider],
                )
            except Exception:
                pass
            else:
                optimized = onnx.load(optimized_path)
                direct_nodes = [
                    node
                    for node in optimized.graph.node
                    if node.domain == "ai.onnx.preview"
                    and node.op_type == "FlexAttention"
                ]
                adapter_accepted = len(direct_nodes) == 1
        if adapter_accepted != case.accepted:
            raise AssertionError(
                f"{provider} adapter disagreed with corpus case {case.name!r}"
            )

    print(f"verified {len(CASES)} modifier cases on Python and {provider}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--ep-library", type=Path)
    args = parser.parse_args()
    verify(args.provider, args.ep_library)


if __name__ == "__main__":
    main()
