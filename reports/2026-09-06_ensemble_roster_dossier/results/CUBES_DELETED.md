# The 20 sample cubes were deleted on 2026-09-16 (50 GB)

`cube_<model>_<composition>_<clamp>/` — the raw `y_pred.npy` / `identifiers.npz` output of each
roster run — were copies of the pipeline's own prediction dirs, kept so the scorer, the pooling test
(M66/M69) and the maps (M74) could read them. All 20 were scored (`score_*.csv`, still here), the
maps are rendered (`maps/`), and the ledger rows M66–M75 are written. The cubes were raw evidence,
not a product, and the disk was full.

To reproduce one: the artifacts are untouched in views-models; `tools/roster_arm_entry.py` re-emits
a cube in ~25 min on the RTX 4070. The body-mean dumps (`bodydump/`, 3 MB) are kept — M70/M75 and
the maps' gate/body rows read those, not the cubes.
