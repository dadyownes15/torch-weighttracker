"""Benchmark the sparse reduction backend against the loop backend.

Usage:
    python sanity_checks/sparse_reduction_benchmark.py --device cuda --json out.json

Every row is measured for both backends on the same model and weights, after a
parity check that the two backends agree.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fixtures_models import CifarResNet20  # noqa: E402
from torch_weighttracker import WeightTracker  # noqa: E402
from torch_weighttracker.calculations import CalcType  # noqa: E402

TRACKERS = (
    "structured_bops",
    "unstructured_sparsity",
    "group_pruning_summary",
    "l2_norm_distribution",
)


def build_model(name: str) -> tuple[nn.Module, torch.Tensor]:
    if name == "resnet20":
        return CifarResNet20(), torch.randn(1, 3, 32, 32)
    if name == "resnet20x4":
        return CifarResNet20(width=64), torch.randn(1, 3, 32, 32)
    if name == "resnet50":
        import torchvision

        return torchvision.models.resnet50(), torch.randn(1, 3, 224, 224)
    raise ValueError(f"Unknown model {name!r}")


def sparsify_(model: nn.Module, seed: int = 0) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                mask = torch.rand(weight.shape, generator=generator) > 0.3
                weight.mul_(mask.to(weight))
                dead = torch.rand(weight.shape[0], generator=generator) < 0.2
                weight[dead.to(weight.device)] = 0


def timed(fn, device: torch.device, *, iters: int, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def run(model_name: str, device: torch.device, iters: int) -> dict:
    torch.manual_seed(0)
    model, example = build_model(model_name)
    model = model.to(device).eval()
    example = example.to(device)
    sparsify_(model)

    owners = {}
    build_seconds = {}
    for backend in ("loop", "sparse"):
        start = time.perf_counter()
        owner = WeightTracker(
            model, example_inputs=example, device=device, reduction_backend=backend
        )
        owner.create_tracker(list(TRACKERS))
        regularizer = owner.create_regularizer("group_lasso")
        owner.get_calculation(CalcType.ACTIVE_UNITS)
        build_seconds[backend] = time.perf_counter() - start
        owners[backend] = (owner, regularizer)

    with torch.no_grad():
        for calc_type in (CalcType.L2_NORM_PR_UNIT, CalcType.ACTIVE_UNITS):
            torch.testing.assert_close(
                owners["sparse"][0].get_calculation(calc_type)(),
                owners["loop"][0].get_calculation(calc_type)(),
                rtol=1e-5,
                atol=1e-6,
            )

    def group_lasso_step(regularizer):
        def step():
            loss = regularizer()
            loss.backward()

        return step

    rows = []
    cases = (
        ("L2_NORM_PR_UNIT", lambda o, r: o.get_calculation(CalcType.L2_NORM_PR_UNIT)),
        ("ACTIVE_UNITS", lambda o, r: o.get_calculation(CalcType.ACTIVE_UNITS)),
        ("track() x4 trackers", lambda o, r: o.track),
        ("group lasso fwd+bwd", lambda o, r: group_lasso_step(r)),
    )
    for label, make in cases:
        timings = {}
        for backend, (owner, regularizer) in owners.items():
            fn = make(owner, regularizer)
            grad = label.startswith("group lasso")
            with torch.enable_grad() if grad else torch.no_grad():
                timings[backend] = timed(fn, device, iters=iters)
        rows.append(
            {
                "case": label,
                "loop_ms": timings["loop"],
                "sparse_ms": timings["sparse"],
                "speedup": timings["loop"] / timings["sparse"],
            }
        )

    return {
        "model": model_name,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else f"CPU ({platform.machine()})"
        ),
        "torch": torch.__version__,
        "params": sum(p.numel() for p in model.parameters()),
        "canonical_groups": len(owners["sparse"][0].canonical_groups),
        "build_seconds": build_seconds,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--models", nargs="+", default=["resnet20"])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    results = [run(name, torch.device(args.device), args.iters) for name in args.models]
    for result in results:
        print(
            f"\n{result['model']} on {result['device_name']} ({result['device']}), "
            f"{result['params']:,} params, torch {result['torch']}"
        )
        print(f"{'case':24s} {'loop ms':>9s} {'sparse ms':>10s} {'speedup':>8s}")
        for row in result["rows"]:
            print(
                f"{row['case']:24s} {row['loop_ms']:9.3f} "
                f"{row['sparse_ms']:10.3f} {row['speedup']:7.1f}x"
            )
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
