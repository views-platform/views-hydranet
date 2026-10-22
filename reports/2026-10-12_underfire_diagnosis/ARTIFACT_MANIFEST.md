# Artifact manifest — the frozen cubes this dossier's results rest on

**Written 2026-10-22.** ⛔ **These files are NOT in git and NOT backed up anywhere.**
`reports/` is gitignored (`.gitignore:203`) and none of them was ever force-added, so they exist on
one disk, in one directory. The `nb` control's **checkpoint has been deleted**, which makes
`d4_result_CONTROL_nb_4acbf4d3.npz` the sole surviving record of the delivered control model.

Every number in `reports/2026-10-22_emit_time_split_method_review.md`, in `RESULTS_LEDGER` M66/M67,
and in the 2026-10-21 emit-time prereg's OUTCOME was measured on these. If they go, those results
become uncheckable — not wrong, but unverifiable, which for this programme is the same thing.

| file | size | sha256 (first 16) | holds |
|---|---|---|---|
| `d4_result_CONTROL_nb_4acbf4d3.npz` | 133 MB | `cf13f2d7e165d7d3…` | ⭐ the **`nb` control**: `mu`, `theta`, `gate` at (4 dropout passes, 49 months 421–469, 64,818 land cells), plus `land_cells`, `months`, `sha`. Checkpoint deleted. |
| `d4_result.npz` | 133 MB | `cf13f2d7e165d7d3…` | ⛔ **BYTE-IDENTICAL to the control cube** — same sha256, different name. Pure duplication; one of the two can go. |
| `d4_result_tnb.npz` | 129 MB | `9d00420e901f0e90…` | the `truncated_nb`-trained arm, same shape |
| `d4_result_tnbNODROP.npz` | 18 MB | `dc175fd5f39fe252…` | the dropout-OFF condition (1 pass × 16), C-128's cost test |
| `p2_result.npz` | 15 MB | `2186c479b8e92dc6…` | the gate-calibration readout (P2) |

**430 MB total, 297 MB after de-duplication.**

## ⛔ Why they are not archived yet

Simon's decision (2026-10-22) was to archive them **outside git**. There is currently nowhere to
archive them *to*:

- no external media mounted (`/media/simon` empty), no network volume;
- the system disk is at **99% — 18 GB free of 938 GB**.

A second copy on the same volume is not an archive, so this manifest records *what* needs saving
and *why* rather than pretending the job is done. ⚠️ The 99%-full disk is independently a live risk
for anything that retrieves artifacts from a pod — this repo already has a scar where a run filled
the disk with a 37 GB wandb cache.

**What a reader should do before trusting a number from this dossier:** check that the file is still
present and its sha256 still matches. If it does not, the number is unverifiable and should be
re-measured or withdrawn, not quoted.
