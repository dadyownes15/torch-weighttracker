# GroupLasso Testing Spec

## Goal

Make `GroupLasso` tests prove the regularizer's contract, not just that the
regularizer can be wired and backpropagated.

The tests should cover:

- exact formula evaluation;
- required-calculation enforcement;
- `StructureTracker` integration;
- gradient values, not only gradient existence;
- zero-unit behavior without NaNs;
- cached/static dependency behavior.

Do not rely on legacy `torch_structure_analyser` tests for this. Add current
package tests under `tests/`.

## Runtime Contract

`GroupLasso.forward()` currently computes:

```python
unit_active_mask = UNIT_ACTIVE_MASK()
active_pr_group = UNITS_TO_GROUP(unit_active_mask)
baseline_group_size = BASELINE_GROUP_SIZES()
group_change_effect = GROUP_CHANGE_EFFECT()
group_sizes = GROUP_SIZES()
l2_norm_pr_unit = L2_NORM_PR_UNIT()

active_params_pr_group = (
    active_pr_group - baseline_group_size
) * group_change_effect
active_params_pr_unit = torch.repeat_interleave(
    active_params_pr_group,
    group_sizes,
)
loss = (active_params_pr_unit * l2_norm_pr_unit).sum()
```

Tests should make this formula explicit. If the sign or naming of
`active_params_pr_group` is later changed, tests should fail loudly instead of
only checking shapes or non-null gradients.

## Test File Shape

Create a dedicated current-package file:

```text
tests/test_group_lasso_regularizer.py
```

This avoids depending on `tests/test_calculation_runtime.py`, which currently
contains unrelated legacy imports and is not a clean collection target.

## Formula-Level Tests

### Exact Dependency-Injection Test

Construct `GroupLasso` directly with simple test calculations:

```text
UNIT_ACTIVE_MASK        -> [1, 0, 1, 1]
UNITS_TO_GROUP(mask)    -> [2, 1]
BASELINE_GROUP_SIZES    -> [3, 1]
GROUP_CHANGE_EFFECT     -> [10, 4]
GROUP_SIZES             -> [3, 1]
L2_NORM_PR_UNIT         -> trainable tensor [5, 7, 11, 13]
```

Expected:

```text
active_params_pr_group = [(2 - 3) * 10, (1 - 1) * 4]
                       = [-10, 0]
active_params_pr_unit  = [-10, -10, -10, 0]
loss                   = -10 * (5 + 7 + 11) + 0 * 13
                       = -230
```

Assertions:

- loss equals `-230`;
- gradient on `L2_NORM_PR_UNIT` equals `[-10, -10, -10, 0]`;
- required calculations are enforced by `GroupLasso({})`.

This is the canonical formula test. It should not use real modules or
canonical groups.

### Repeat-Interleave Boundary Test

Use three groups with uneven sizes:

```text
GROUP_SIZES = [1, 3, 2]
active_params_pr_group = [2, -5, 7]
```

Expected repeated coefficients:

```text
[2, -5, -5, -5, 7, 7]
```

This catches off-by-one and group-order errors in the formula.

## StructureTracker Integration Tests

### Actual-Value Case Matrix

Add table-driven tests where each case defines:

```text
name
model factory
group/canonicalization factory
weight initialization
expected UNIT_ACTIVE_MASK
expected UNITS_TO_GROUP(UNIT_ACTIVE_MASK)
expected BASELINE_GROUP_SIZES
expected GROUP_CHANGE_EFFECT
expected GROUP_SIZES
expected L2_NORM_PR_UNIT
expected GroupLasso loss
expected selected gradients
```

These tests should use real modules and real calculation outputs. They should
not compare `GroupLasso` against itself; expected tensors must be written out by
hand.

Minimum case matrix:

| Case | Model Type | Unit Semantics | Must Verify |
| --- | --- | --- | --- |
| `linear_chain_channel_units` | `Linear -> Linear` | plain channel units | hidden output/input coupling, zero hidden unit, exact loss and gradients |
| `conv_bn_channel_units` | `Conv2d(groups=1) -> BatchNorm2d -> Conv2d` | conv output + BN feature + next conv input | conv/BN/conv coupled channel group, exact `GROUP_CHANGE_EFFECT`, exact L2 values |
| `residual_branch_channel_units` | small residual block with shortcut conv | shared output units across main and shortcut paths | all linked members contribute to `L2_NORM_PR_UNIT`; active mask is unit-level, not per-member |
| `qkv_proj_channel_units` | fused `qkv: Linear(E, 3E)` + `proj: Linear(E, E)` | attention channel units | fused QKV member contributes semantic channel L2; proj input member is linked |
| `qkv_proj_head_units` | same as above | attention head units | group length equals `num_heads`; one head's expected value aggregates its head-dim slice |
| `qkv_proj_head_dim_units` | same as above | attention head-dim units | group length equals `head_dim`; one head-dim unit aggregates across heads and Q/K/V |

If any case is not currently supported, mark it as `xfail` with the exact reason
and keep the expected values in the test body. Do not silently omit the case.

### Per-Case Expected Values

For every case, assert the intermediate calculation values before asserting the
regularizer loss:

```python
unit_active_mask = tracker.get_calculation(CalcType.UNIT_ACTIVE_MASK)()
active_pr_group = tracker.get_calculation(CalcType.UNITS_TO_GROUP)(
    unit_active_mask,
)
baseline_group_sizes = tracker.get_calculation(CalcType.BASELINE_GROUP_SIZES)()
group_change_effect = tracker.get_calculation(CalcType.GROUP_CHANGE_EFFECT)()
group_sizes = tracker.get_calculation(CalcType.GROUP_SIZES)()
l2_norm_pr_unit = tracker.get_calculation(CalcType.L2_NORM_PR_UNIT)()
loss = tracker.create_regularizer(RegularizerType.GROUP_LASSO)()
```

Then assert all of them against explicit expected tensors. This makes failures
diagnosable: a bad loss should tell us whether the error came from active-mask
semantics, group aggregation, structural counts, L2 reductions, or the final
formula.

### Linear Expected-Value Baseline

The existing `TinyLinearChain` fixture should be the first exact-value case.
It should assert:

```text
UNIT_ACTIVE_MASK      = [1, 0, 1, 1]
UNITS_TO_GROUP(mask)  = [2, 1]
BASELINE_GROUP_SIZES  = [3, 1]
GROUP_CHANGE_EFFECT   = [3, 3]
GROUP_SIZES           = [3, 1]
L2_NORM_PR_UNIT       = [5, 0, sqrt(13) + 6, sqrt(52)]
active_params_group   = [-3, 0]
active_params_unit    = [-3, -3, -3, 0]
loss                  = -3 * (5 + 0 + sqrt(13) + 6)
```

The gradient assertion should include exact per-weight values for at least one
nonzero unit and the zero unit. The zero hidden unit must not produce NaNs.

### Conv/BN Expected-Value Baseline

Use a tiny deterministic conv block:

```text
conv1: Conv2d(1 -> 2, kernel_size=1, bias=False)
bn1:   BatchNorm2d(2, affine=True)
conv2: Conv2d(2 -> 1, kernel_size=1, bias=False)
```

Create one group linking:

```text
conv1 prune_out_channels
bn1 prune_out_channels / feature
conv2 prune_in_channels
```

Set weights so channel 0 is active and channel 1 is zero across all linked
members. Expected values must include:

```text
UNIT_ACTIVE_MASK      = [1, 0]
UNITS_TO_GROUP(mask)  = [1]
BASELINE_GROUP_SIZES  = [2]
GROUP_SIZES           = [2]
```

`GROUP_CHANGE_EFFECT` must be asserted from the representative unit's parameter
count, not guessed from module count. Write the exact expected number in the
test after choosing the concrete tensor shapes.

### Attention Expected-Value Baselines

For qkv/proj cases, use small dimensions:

```text
embed_dim = 4
num_heads = 2
head_dim = 2
qkv:  Linear(4 -> 12, bias=False)
proj: Linear(4 -> 4, bias=False)
```

Use deterministic weights, preferably all ones with selected semantic units
zeroed. For each mode, assert:

- canonical group length;
- `UNIT_ACTIVE_MASK`;
- `GROUP_SIZES`;
- `L2_NORM_PR_UNIT`;
- final loss;
- gradients on both `qkv.weight` and `proj.weight`.

The three attention modes must be separate tests because their expected L2
aggregation differs:

- `channel`: one unit corresponds to one embed channel, with Q/K/V rows linked;
- `head`: one unit corresponds to a full head, aggregating `head_dim` channels;
- `head_dim`: one unit corresponds to one dimension inside every head,
  aggregating across heads.

Use the existing canonicalization knobs:

```text
channel:  prune_dim=False, prune_num_heads=False
head:     prune_num_heads=True
head_dim: prune_dim=True
```

Do not only check that the tensors have the right length. The expected values
must prove that the semantic aggregation is correct.

### Exact Tiny Linear Chain Test

Use the existing `_model_and_groups()` fixture from
`tests/test_calculation_specs.py`.

Expected dependencies from that fixture:

```text
UNIT_ACTIVE_MASK      = [1, 0, 1, 1]
UNITS_TO_GROUP(mask)  = [2, 1]
BASELINE_GROUP_SIZES  = [3, 1]
GROUP_CHANGE_EFFECT   = [3, 3]
GROUP_SIZES           = [3, 1]
```

The test should create:

```python
regularizer = tracker.create_regularizer(RegularizerType.GROUP_LASSO)
loss = regularizer()
```

Assertions:

- `loss` equals the manual formula from the dependency tensors;
- `regularizer` is appended to `tracker.regularizers`;
- each required calculation exists in `tracker.calculations`;
- `BASELINE_GROUP_SIZES`, `GROUP_CHANGE_EFFECT`, and `GROUP_SIZES` are
  `CachedCalculation` instances;
- dynamic calculations are not cached.

### Gradient Value Test

Use a tiny fixture with no zero-weight unit to test exact gradient values without
the zero-norm edge case. The expected gradient for each weight slice should be:

```text
coefficient_for_unit * d(L2(slice)) / d(slice)
```

Assertions:

- every involved parameter has a gradient;
- gradients match explicit hand-computed tensors;
- no unrelated parameter receives a gradient.

### Zero-Unit Gradient Test

Add a separate test for a group with a fully zero unit.

The expected behavior must be explicit:

- no NaN gradients;
- zeroed slices receive zero gradient, unless the implementation deliberately
  uses an epsilon-smoothed L2 norm and documents the resulting gradient.

This test is important because `sqrt(sum(w**2))` has an undefined derivative at
zero. A test that only checks `grad is not None` can miss NaNs.

## Cache and Mutation Tests

### Baseline Size Cache

Create a tracker, fetch `BASELINE_GROUP_SIZES`, then mutate weights so a unit
becomes zero.

Assertions:

- `BASELINE_GROUP_SIZES()` is unchanged;
- `UNIT_ACTIVE_MASK()` changes;
- `GroupLasso()` reflects the changed active mask.

### Group Change Effect Cache

Fetch `GROUP_CHANGE_EFFECT`, mutate weights, and verify it remains unchanged.
It is structural and must not depend on current values.

## Attention/QKV Tests

GroupLasso should inherit the canonical-group attention semantics already used
by `L2_NORM_PR_UNIT`.

Add a focused timm-style fixture with:

```text
qkv:  Linear(E -> 3E)
proj: Linear(E -> E)
num_heads = H
prune_dim = True
```

Use a canonical group shaped like:

```text
members for attn.proj:prune_in_channels:head_dim:
  - attn.proj
  - attn.qkv
```

Assertions:

- group length is `head_dim`;
- `L2_NORM_PR_UNIT` has one value per head-dim unit;
- `GroupLasso` loss equals the direct formula from the calculation outputs;
- gradients flow into both `qkv.weight` and `proj.weight`.

Do not test `nn.MultiheadAttention` parent-module GroupLasso here unless MHA
canonicalization support is being changed in the same implementation pass.

## Test Commands

Focused validation:

```bash
python -m pytest tests/test_group_lasso_regularizer.py -q
python -m pytest tests/test_structure_tracker_unit_extensive.py::test_create_regularizer_wires_group_lasso_and_keeps_gradients -q
```

Broader current-package validation:

```bash
python -m pytest tests/test_calculation_specs.py tests/test_structure_tracker_unit_extensive.py -q
```

Avoid using `tests/test_calculation_runtime.py` as the primary target until its
unrelated legacy import issue is resolved.
