# Structured Unit Sum timm Benchmark

This folder benchmarks `StructuredUnitSum` against an intentionally naive
runner on a timm model. Both implementations use the same reducer plan and
produce the same tensor; the naive runner rebuilds the accumulator and
destination index tensors every call, while `StructuredUnitSum` reuses
precompiled buffers.

Run the default ViT benchmark from the repo root:

```bash
python benchmarks/structured_unit_sum/benchmark_timm_structured_unit_sum.py
```

The script writes:

- `benchmarks/structured_unit_sum/results/latest.json`
- `benchmarks/structured_unit_sum/results/latest.csv`

Open `structured_unit_sum_benchmark.ipynb` in this folder to view the latest
table and chart, or rerun the benchmark from the first notebook cell.

Useful options:

```bash
python benchmarks/structured_unit_sum/benchmark_timm_structured_unit_sum.py \
  --model vit_tiny_patch16_224 \
  --img-size 32 \
  --iterations 200 \
  --repeats 7 \
  --threads 1
```

For attention-specific semantic reductions:

```bash
python benchmarks/structured_unit_sum/benchmark_timm_structured_unit_sum.py \
  --attention-reduction heads
```
