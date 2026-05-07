from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def install_local_structracker_packages() -> None:
    """Import local submodules without depending on the public API exports."""

    package_root = REPO_ROOT / "torch_structracker"
    package = types.ModuleType("torch_structracker")
    package.__file__ = str(package_root / "__init__.py")
    package.__path__ = [str(package_root)]
    package.__package__ = "torch_structracker"
    sys.modules["torch_structracker"] = package

    calculations_root = package_root / "calculations"
    calculations_package = types.ModuleType("torch_structracker.calculations")
    calculations_package.__file__ = str(calculations_root / "__init__.py")
    calculations_package.__path__ = [str(calculations_root)]
    calculations_package.__package__ = "torch_structracker.calculations"
    sys.modules["torch_structracker.calculations"] = calculations_package


install_local_structracker_packages()

try:
    import timm
except ImportError as exc:  # pragma: no cover - exercised only without timm.
    raise SystemExit(
        "This benchmark requires timm. Install the dev dependencies or run "
        "`pip install timm`."
    ) from exc

from torch_structracker.calculations.structured_unit_sum import StructuredUnitSum
from torch_structracker.operations import WeightOperationType
from torch_structracker.reducer_plan import (
    ReducerPlan,
    compile_reducer_plan_from_groups,
    validate_reducer_plan,
)
from torch_structracker.torch_pruning.dependency import DependencyGraph

try:
    from .naive_impl import NaiveStructuredUnitSum
except ImportError:  # pragma: no cover - direct script execution.
    from naive_impl import NaiveStructuredUnitSum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark torch-structracker StructuredUnitSum against a naive "
            "allocation-heavy runner on a timm model."
        )
    )
    parser.add_argument("--model", default="vit_tiny_patch16_224")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--attention-reduction",
        choices=("none", "head-dim", "heads"),
        default="none",
        help=(
            "Optional QKV semantic reduction mode for attention groups. "
            "The default keeps the dependency-group unit layout."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional torch CPU thread count for more reproducible CPU timings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "latest.json",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "latest.csv",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    if (
        device.type == "mps"
        and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available())
    ):
        raise SystemExit("MPS was requested but is not available.")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def create_model(
    model_name: str,
    img_size: int,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    model = timm.create_model(
        model_name,
        pretrained=False,
        img_size=img_size,
        num_classes=num_classes,
    )
    model.eval()
    return model.to(device=device)


def qkv_num_heads(model: nn.Module) -> dict[nn.Module, int]:
    num_heads: dict[nn.Module, int] = {}
    for module in model.modules():
        qkv = getattr(module, "qkv", None)
        heads = getattr(module, "num_heads", None)
        if isinstance(qkv, nn.Linear) and heads is not None:
            num_heads[qkv] = int(heads)
    return num_heads


def build_plan(
    model: nn.Module,
    example_inputs: torch.Tensor,
    attention_reduction: str,
) -> tuple[ReducerPlan, int, float]:
    start = time.perf_counter()
    graph = DependencyGraph().build_dependency(
        model=model,
        example_inputs=example_inputs,
    )
    groups = list(graph.get_all_groups(root_module_types=[nn.Linear]))
    plan = compile_reducer_plan_from_groups(
        groups,
        operation_type=WeightOperationType.SUM,
        num_heads=qkv_num_heads(model),
        prune_dim=attention_reduction == "head-dim",
        prune_num_heads=attention_reduction == "heads",
    )
    validate_reducer_plan(plan)
    elapsed_seconds = time.perf_counter() - start
    return plan, len(groups), elapsed_seconds


def time_callable(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, float | list[float]]:
    if warmup < 0 or iterations <= 0 or repeats <= 0:
        raise ValueError("warmup must be >= 0, iterations and repeats must be > 0.")

    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        synchronize(device)

        per_iteration_seconds = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iterations):
                fn()
            synchronize(device)
            elapsed = time.perf_counter() - start
            per_iteration_seconds.append(elapsed / iterations)

    median_seconds = statistics.median(per_iteration_seconds)
    mean_seconds = statistics.mean(per_iteration_seconds)
    stdev_seconds = (
        statistics.stdev(per_iteration_seconds)
        if len(per_iteration_seconds) > 1
        else 0.0
    )
    return {
        "median_ms": median_seconds * 1000.0,
        "mean_ms": mean_seconds * 1000.0,
        "min_ms": min(per_iteration_seconds) * 1000.0,
        "max_ms": max(per_iteration_seconds) * 1000.0,
        "stdev_ms": stdev_seconds * 1000.0,
        "samples_ms": [value * 1000.0 for value in per_iteration_seconds],
    }


def max_abs_diff(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.detach() - second.detach()).abs().max().cpu().item())


def write_result(result: dict, output_path: Path, csv_output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "implementation",
                "median_ms",
                "mean_ms",
                "min_ms",
                "max_ms",
                "stdev_ms",
            ],
        )
        writer.writeheader()
        for implementation in ("structured", "naive"):
            row = {"implementation": implementation}
            row.update(
                {
                    key: result["timing"][implementation][key]
                    for key in (
                        "median_ms",
                        "mean_ms",
                        "min_ms",
                        "max_ms",
                        "stdev_ms",
                    )
                }
            )
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    device = resolve_device(args.device)
    example_inputs = torch.ones(
        args.batch_size,
        3,
        args.img_size,
        args.img_size,
        device=device,
    )
    model = create_model(
        model_name=args.model,
        img_size=args.img_size,
        num_classes=args.num_classes,
        device=device,
    )
    plan, group_count, plan_build_seconds = build_plan(
        model=model,
        example_inputs=example_inputs,
        attention_reduction=args.attention_reduction,
    )

    structured = StructuredUnitSum(plan)
    naive = NaiveStructuredUnitSum(plan)

    with torch.inference_mode():
        structured_result = structured().clone()
        naive_result = naive()
    torch.testing.assert_close(structured_result, naive_result)

    structured_timing = time_callable(
        structured,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    naive_timing = time_callable(
        naive,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )

    speedup = naive_timing["median_ms"] / structured_timing["median_ms"]
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "attention_reduction": args.attention_reduction,
            "torch_threads": torch.get_num_threads(),
        },
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "python": sys.version.split()[0],
        },
        "model": {
            "name": args.model,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "num_classes": args.num_classes,
        },
        "plan": {
            "dependency_groups": group_count,
            "mappings": len(plan.mappings),
            "output_length": plan.output_length,
            "plan_build_ms": plan_build_seconds * 1000.0,
            "naive_destination_tensor_allocations_per_call": (
                naive.destination_tensor_allocations_per_call
            ),
            "structured_destination_tensor_allocations_per_call": 0,
        },
        "correctness": {
            "max_abs_diff": max_abs_diff(structured_result, naive_result),
        },
        "timing": {
            "structured": structured_timing,
            "naive": naive_timing,
            "speedup_x": speedup,
            "median_ms_saved": (
                naive_timing["median_ms"] - structured_timing["median_ms"]
            ),
        },
    }

    write_result(result, args.output, args.csv_output)

    print(
        "StructuredUnitSum median: "
        f"{structured_timing['median_ms']:.4f} ms | "
        "naive median: "
        f"{naive_timing['median_ms']:.4f} ms | "
        f"speedup: {speedup:.2f}x"
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
