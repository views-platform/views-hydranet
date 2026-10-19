# T0 / T2 / T3 — measurement scripts for views-hydranet#396

Pre-registered in `../2026-10-12_underfire_diagnosis_prereg.md`. **Written 2026-10-12, before the
artifact exists** — the four predictions were committed first (`ea6f114`), so writing the code
cannot bias the result.

## Status

⛔ **Cannot run yet.** All three need a global-land-trained `violet_visitor`, which does not exist
anywhere (views-models#549 explains why the 2026-09-28 one is gone). Blocked on views-models#550.

T1 is already **complete** and needed no vehicle — see the prereg's outcome section.

## Run order, once the artifact lands

```
ART=.../calibration_model_<ts>.pt           # all THREE files must be present:
                                            #   .pt  .pt.config.json  .pt.sha256
conda run -n views-hydranet-env python t2_gradient_partition.py "$ART"   # decides #394
conda run -n views-hydranet-env python t3_scoring.py            "$ART"
conda run -n views-hydranet-env python t0_gate_vs_body.py       "$ART"
```

T2 first: it is the only real test of views-hydranet#394's mechanism, and #394 is the intervention
this whole diagnosis gates.

The first run performs one fresh datafactory pull of all three targets at `REGION="land"`, months
121–504, cached to `pull_land_all3_121_504.parquet` (~5 s, ~400 MB).

## What each one measures

| script | question | fires |
|---|---|---|
| `t2_gradient_partition.py` | Does #394's non-land dilution reach the **gradient**? #394 asserts 75% of *terms*; nobody has shown the gradient follows. One forward/backward, partitioned non-land / zero-land / active-land by loss value **and** gradient norm. | **P3**, **F3** |
| `t3_scoring.py` | Does the objective **reward** timidity? | **P4**, **F4** |
| `t0_gate_vs_body.py` | Is the composed zero mass from `(1 − g)` or from `P_body(0)`? | **P2**, **F5** |

`vehicle.py` is shared: artifact loading with the F6 gate, the fresh pull, and grid placement.

## Two things worth knowing before reading the numbers

**T3 runs two comparisons, and the second is not in the prereg.** The pre-registered comparison
(model vs an honest NB fitted to observed history) is **confounded**: the honest fit is a per-cell
constant while the model is conditional on inputs, so a model win could be legitimate conditioning
rather than rewarded timidity. So T3 also scores the model against a **mean-preserving
de-timidified version of itself** — `mu` held fixed, `theta` raised until `P(Y=0)` equals the
observed zero rate. Same mean, same conditioning, strictly less zero mass. If the timid one still
wins, the objective prefers emptiness and there is no confound left. The NB's floor at fixed mean is
`exp(-mu)` (the Poisson limit); cells whose observed zero rate sits below that floor are reported
separately rather than forced.

**⛔ Region provenance is NOT verifiable from the artifact.** Found by smoke-testing the F6 guard
against the local artifacts: *all four pass the config check* despite being africa-trained, because
the sidecar records no region or grid size and the network is fully convolutional, so neither the
config nor the state-dict shapes discriminate. Config identity is therefore **necessary and not
sufficient**. Handled three ways, and the limit is stated rather than hidden:

1. the four known africa artifacts are **denied by sha256** (`vehicle.py:KNOWN_AFRICA_SHA256`),
2. `load_vehicle` prints a loud notice that region provenance is the operator's assertion,
3. that assertion rests on `pod_run_model.sh:279-282`, which **refuses to start** unless
   `REGION == 'land'`.

## Grid placement

`row = (pg−1)//720 + 1`, `col = (pg−1)%720 + 1` on the 360×720 PRIO-GRID raster. **Validated
2026-10-12** against the production raw file's own `row`/`col` columns on all 13,110 africa cells —
exact match on both axes.

## Conventions honoured

Targets go to the loss in **log1p space**; the family's `nll` recovers raw counts itself via
`count_target_bridge.to_raw_counts`. Parameter slicing mirrors `training_engine.py:686` exactly
(ADR-067 / C-207). The loss is built from the repo's own `FamilyLoss` / `get_family`, never a
reimplementation. Non-land cells are left at zero, which is what the production raster does
(`config_queryset.py:27`). `disk_guard` is **not** consulted — postmortem §2.5 — and T2 records
`torch.cuda.max_memory_allocated()` instead.
