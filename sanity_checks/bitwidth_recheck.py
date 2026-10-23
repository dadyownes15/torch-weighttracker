"""Sanity check: BOPs follow bitwidth changes made during training.

Usage:
    python sanity_checks/bitwidth_recheck.py --device cpu --json out.json

A tiny CNN is "trained" for a few steps with a `structured_bops` tracker. The
model mimics a mixed-precision setup:

    conv1  weight quantizer on `parametrizations.weight` (CoDeQ-style)
    conv2  input activations quantized by a separate module in front of it
    fc     not quantized (32/32)

Halfway through training the quantizers lower conv1's weights from 8 to 4 bits
and conv2's input activations from 8 to 6 bits. The script checks that the next
`track()` reports BOPs scaled by exactly 4/8 and 6/8 for those layers, that fc
is unchanged, and that the total is the sum. It exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_weighttracker import WeightTracker  # noqa: E402

TRACKER = "structured_bops"


class WeightQuantizer(nn.Module):
    """Stands in for a CoDeQ quantizer: `get_bitwidth()` returns a CPU tensor."""

    def __init__(self, bits: int) -> None:
        super().__init__()
        self.register_buffer("bits", torch.tensor(float(bits)))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight

    def get_bitwidth(self) -> torch.Tensor:
        return self.bits.detach().clone().cpu()


class WeightQuantParametrization(nn.Module):
    def __init__(self, bits: int) -> None:
        super().__init__()
        self.quantizer = WeightQuantizer(bits)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return self.quantizer(weight)


class ActivationQuantizer(nn.Module):
    """A separate activation quantizer module, applied before a layer."""

    def __init__(self, bits: int) -> None:
        super().__init__()
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.act_quant = ActivationQuantizer(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10, bias=False)
        parametrize.register_parametrization(
            self.conv1, "weight", WeightQuantParametrization(8)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(self.act_quant(x)))
        return self.fc(self.pool(x).flatten(1))


def weight_bitwidth_fn(name: str, module: nn.Module):
    """Weight bits: read from the quantizer parametrization on `weight`."""
    if parametrize.is_parametrized(module, "weight"):
        for parametrization in module.parametrizations.weight:
            return parametrization.quantizer.get_bitwidth()
    return 32


def make_activation_bitwidth_fn(model: TinyCNN):
    """Input-activation bits: each layer maps to the quantizer that feeds it."""
    input_quantizers = {"conv2": model.act_quant}

    def activation_bitwidth_fn(name: str, module: nn.Module):
        quantizer = input_quantizers.get(name)
        return 32 if quantizer is None else quantizer.bits

    return activation_bitwidth_fn


def layer_bops(metrics: dict) -> dict[str, float]:
    return {name: values["bops"] for name, values in metrics["modules"].items()}


def mean_track_ms(tracker, iters: int, device: torch.device) -> float:
    for _ in range(5):
        tracker.track()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        tracker.track()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    model = TinyCNN().to(device)
    example = torch.randn(1, 3, 16, 16, device=device)
    owner = WeightTracker(
        model,
        example,
        activation_bitwidth_fn=make_activation_bitwidth_fn(model),
        weight_bitwidth_fn=weight_bitwidth_fn,
    )
    tracker = owner.create_tracker(
        TRACKER, log_total_bops=True, log_layerwise_stats=True
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    change_step = args.steps // 2

    history = []
    before = after = None
    for step in range(args.steps):
        if step == change_step:
            # What the quantizers do during mixed-precision training.
            with torch.no_grad():
                model.conv1.parametrizations.weight[0].quantizer.bits.fill_(4)
            model.act_quant.bits = 6
            after = layer_bops(tracker.track()[TRACKER])

        x = torch.randn(8, 3, 16, 16, device=device)
        y = torch.randint(0, 10, (8,), device=device)
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()

        metrics = tracker.track()[TRACKER]
        if step == change_step - 1:
            before = layer_bops(metrics)
        history.append({"step": step, "bops": metrics["bops"]})

    expected_ratio = {"conv1": 4 / 8, "conv2": 6 / 8, "fc": 1.0}
    views = {layer.name: layer for layer in owner.view_bitwidths().layers}

    print(
        f"{'layer':<6} {'bits (a/w)':>10} {'BOPs before':>14} {'BOPs after':>14} "
        f"{'ratio':>7} {'expected':>8}"
    )
    failures = []
    rows = []
    for name, ratio in expected_ratio.items():
        actual = after[name] / before[name]
        ok = abs(actual - ratio) < 1e-6
        bits = f"{views[name].activation_bits}/{views[name].weight_bits}"
        print(
            f"{name:<6} {bits:>10} {before[name]:>14,.0f} {after[name]:>14,.0f} "
            f"{actual:>7.3f} {ratio:>8.3f} {'ok' if ok else 'MISMATCH'}"
        )
        rows.append(
            {
                "layer": name,
                "before": before[name],
                "after": after[name],
                "ratio": actual,
                "expected": ratio,
            }
        )
        if not ok:
            failures.append(f"{name}: ratio {actual:.6f}, expected {ratio:.6f}")

    total = tracker.track()[TRACKER]["bops"]
    if abs(total - sum(after.values())) > 1e-6 * total:
        failures.append(f"total {total} != sum of layers {sum(after.values())}")

    plain = WeightTracker(model, example).create_tracker(
        TRACKER, log_total_bops=True, log_layerwise_stats=True
    )
    timing = {
        "track_ms_with_bitwidth_fns": mean_track_ms(tracker, args.iters, device),
        "track_ms_without_bitwidth_fns": mean_track_ms(plain, args.iters, device),
    }
    print(
        f"\ntrack(): {timing['track_ms_with_bitwidth_fns']:.3f} ms with bitwidth fns, "
        f"{timing['track_ms_without_bitwidth_fns']:.3f} ms without "
        f"({args.iters} calls, {device})"
    )

    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "device": str(device),
                    "layers": rows,
                    "history": history,
                    "timing": timing,
                    "failures": failures,
                },
                indent=2,
            )
        )

    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nOK: BOPs followed the bitwidth changes.")


if __name__ == "__main__":
    main()
