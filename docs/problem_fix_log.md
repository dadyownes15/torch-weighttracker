# torch_structracker Problem And Fix Log

This document tracks the concrete problems found while adding extensive unit and
end-to-end coverage for `torch_structracker`. The outdated
`torch_structure_analyser` tree is intentionally out of scope for this pass.

## Scope

The test work targets the current package:

- `torch_structracker.structure_tracker.StructureTracker`
- calculation specs and runtime calculations
- structured BOP/module-axis plumbing
- bitrate extraction
- dependency-group pruning behavior exercised through the vendored
  `torch_structracker.torch_pruning` dependency graph

The main correctness target is that active structured-unit accounting lines up
with physically pruned PyTorch modules, and that selected pruned module FLOP
counts match `fvcore.nn.FlopCountAnalysis`.

## Fixed Problems

### 1. Test Runner Blocked By Merge Conflict Markers

Status: fixed.

Problem:

`pyproject.toml` contained unresolved merge conflict markers. This made pytest
stop before collecting tests:

```text
ERROR: pyproject.toml: Invalid statement
```

Fix:

`pyproject.toml` was rewritten as valid TOML for the current package. The package
discovery now targets `torch_structracker*` only, matching the current code under
test.

Verification:

```bash
python -m pytest tests/test_calculation_specs.py tests/test_calculation_runtime.py tests/test_param_unit_calculator.py tests/test_linear_weight_sum_specs.py tests/test_mha_operations.py tests/test_reduction_plan_builder.py tests/test_module_bitrate_extractor.py tests/test_structure_tracker_unit_extensive.py tests/test_structracker_fvcore_e2e.py -q
```

Result: `73 passed`.

### 2. README Still Described Conflicting Package Identities

Status: fixed.

Problem:

`README.md` also contained merge conflict markers and mixed two package
identities. That made the current package contract unclear.

Fix:

The README now describes `torch-structracker` only and uses a smoke test that
imports `torch_structracker.structure_tracker.StructureTracker`.

Verification:

Conflict markers were checked with:

```bash
rg '(<{7}|>{7}|={7})' pyproject.toml README.md torch_structracker tests docs -n
```

Result: no conflict markers in the checked current-package paths.

### 3. Package Lacked An Explicit Initializer

Status: fixed.

Problem:

`torch_structracker` had no `__init__.py`. Direct submodule imports worked from
the repo, but relying on namespace-package behavior is fragile for a library
surface.

Fix:

Added `torch_structracker/__init__.py` exporting `StructureTracker` and
`__version__`.

Verification:

`tests/test_structure_tracker_unit_extensive.py` includes:

- `test_package_exports_structure_tracker`

### 4. MultiheadAttention Was Missing From Weighted Module Discovery

Status: fixed.

Problem:

`ModuleBitrateExtractor.weighted_modules()` only considered modules with a
`.weight` tensor. `nn.MultiheadAttention` stores its Q/K/V projection weights as
`in_proj_weight` or separate Q/K/V projection parameters, so the attention parent
was skipped.

Impact:

Canonical attention groups can contain the `nn.MultiheadAttention` module, but
the shared weighted-module order for `UNITS_TO_MODULE_AXIS` and
`BITRATE_PR_MODULE` did not include that parent. This meant attention QKV work
could be silently absent from structured BOP accounting.

Fix:

`ModuleBitrateExtractor` now resolves weighted tensors through:

- `module.weight`
- `nn.MultiheadAttention.in_proj_weight`
- `nn.MultiheadAttention.q_proj_weight` for separate-QKV attention

The same resolver is used by `bind()`, `_device_for()`, and
`weighted_modules()`.

Verification:

`tests/test_module_bitrate_extractor.py` includes:

- `test_extractor_lists_multihead_attention_parent_and_children`
- `test_extractor_binds_multihead_attention_projection_bitrates`

### 5. StructureTracker Factory Wiring Needed Direct Unit Coverage

Status: fixed.

Problem:

The suite covered many calculation primitives, but it did not directly exercise
several `StructureTracker` integration contracts:

- building groups from `example_inputs` and `root_module_types`
- rejecting partial dependency-build arguments
- respecting `ignored_layers`
- preserving requested calculation dtype/device
- wiring `TrackerType.STRUCTURED_BOPS`
- wiring `RegularizerType.GROUP_LASSO`
- preserving cached baseline calculations after weight mutation

Fix:

Added `tests/test_structure_tracker_unit_extensive.py`.

Verification:

New unit tests include:

- `test_dependency_build_requires_example_inputs_and_root_types_together`
- `test_structure_tracker_builds_groups_from_example_inputs`
- `test_structure_tracker_removes_ignored_layer_members_from_groups`
- `test_calculation_device_and_dtype_are_applied_to_pipeline_outputs`
- `test_cached_constant_calculations_keep_their_baseline_after_weight_changes`
- `test_create_tracker_wires_structured_bops_from_required_calculations`
- `test_create_regularizer_wires_group_lasso_and_keeps_gradients`

### 6. No Current-Package fvcore Parity Tests For Pruned ResNet Modules

Status: fixed.

Problem:

The current package did not have an end-to-end test proving that active
structured-unit accounting agrees with physically pruned residual CNN modules
and `fvcore.nn.FlopCountAnalysis`.

Fix:

Added a tiny ResNet-style classifier with a residual downsample path. The test:

1. Builds dependency groups with `StructureTracker`.
2. Zeros two units in the residual/downsample output group.
3. Checks `UNIT_ACTIVE_MASK` and `UNITS_TO_MODULE_AXIS` before pruning.
4. Physically prunes the same dependency group.
5. Compares pruned module FLOPs from `fvcore` against hand-computed Conv/Linear
   formulas.

Verification:

`tests/test_structracker_fvcore_e2e.py` includes:

- `test_resnet_residual_prune_axis_counts_match_pruned_fvcore_modules`

Modules compared:

- `block2.downsample.0`
- `block2.conv2`
- `head`

### 7. No Current-Package fvcore Parity Tests For Transformer MLP Pruning

Status: fixed.

Problem:

Transformer-like channel pruning was not covered end to end against `fvcore`.

Fix:

Added an E2E test using `TinyTransformerClassifier`. The test prunes the MLP
hidden dimension and verifies:

- `mlp_in` active input/output axis counts
- `mlp_out` active input/output axis counts
- physical module dimensions after pruning
- per-module `fvcore` counts for `mlp_in`, `mlp_out`, and `head`

Verification:

`tests/test_structracker_fvcore_e2e.py` includes:

- `test_transformer_mlp_prune_axis_counts_match_pruned_fvcore_modules`

### 8. No Current-Package fvcore Parity Tests For Transformer Attention Pruning

Status: fixed.

Problem:

Attention-head structured accounting was not compared against a physically
pruned transformer path.

Fix:

Added an E2E attention-head test using `TinyTransformerClassifier` with
`nn.MultiheadAttention`. The test:

1. Builds head-level canonical groups with `prune_num_heads=True`.
2. Zeros one attention head dependency unit.
3. Verifies semantic active-head axis counts for attention and downstream
   Linear layers.
4. Physically prunes the dependency group.
5. Compares selected pruned module FLOPs with `fvcore`.

Verification:

`tests/test_structracker_fvcore_e2e.py` includes:

- `test_transformer_attention_head_prune_matches_pruned_fvcore_modules`

Modules compared:

- `attn`
- `mlp_in`
- `mlp_out`
- `head`

### 9. New Attention Test Helper Initially Repeated QKV Indices Incorrectly

Status: fixed before landing the final test run.

Problem:

The new E2E helper for zeroing attention slices initially mutated the input
index list while building repeated Q/K/V row indices. That produced invalid
indices for `in_proj_weight`.

Fix:

The helper now builds repeated Q/K/V rows from a copy of the input indices.

Verification:

The failing attention parity test now passes individually and as part of the
targeted suite.

### 10. Default pytest Collection Included Removed Legacy Tests

Status: fixed.

Problem:

After scoping packaging and tests to `torch_structracker`, a plain
`python -m pytest -q` still attempted to collect four legacy test files that
imported `torch_structure_analyser`. That package tree is outdated and was
removed from the active package scope, so collection failed before the current
suite could run.

Fix:

The default pytest configuration now ignores the four legacy test modules:

- `tests/test_attention_axes_e2e.py`
- `tests/test_attention_axes_unit.py`
- `tests/test_model_structured_sparsity.py`
- `tests/test_sparsity_controller.py`

This keeps `pytest` focused on the current `torch_structracker` package while
leaving the legacy tests in the repo for any future archival/migration decision.

Verification:

```bash
python -m pytest -q
```

Result after this fix: `73 passed`; the current-package suite is collected and
run while legacy `torch_structure_analyser` tests are ignored.

## Verification Summary

Focused new/changed coverage:

```bash
python -m pytest tests/test_module_bitrate_extractor.py tests/test_structure_tracker_unit_extensive.py tests/test_structracker_fvcore_e2e.py -q
```

Result:

```text
20 passed
```

Current-package non-legacy suite:

```bash
python -m pytest tests/test_calculation_specs.py tests/test_calculation_runtime.py tests/test_param_unit_calculator.py tests/test_linear_weight_sum_specs.py tests/test_mha_operations.py tests/test_reduction_plan_builder.py tests/test_module_bitrate_extractor.py tests/test_structure_tracker_unit_extensive.py tests/test_structracker_fvcore_e2e.py -q
```

Result:

```text
73 passed
```

## Current Open Problems

No current-package problem found during this pass is intentionally left unfixed
in the code or tests added here.

The outdated `torch_structure_analyser` tree and tests that import it were not
used as the basis for new coverage, per instruction.
