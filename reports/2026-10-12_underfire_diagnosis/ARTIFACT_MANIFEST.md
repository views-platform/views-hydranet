# Artifact manifest — the frozen cubes this dossier's results rest on

**Written 2026-10-22. ✅ ARCHIVED the same day — see §Archive below.**

⚠️ They are still NOT in git, and git is not where they belong.
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
| `d4_result_tnb.npz` | 129 MB | `9d00420e901f0e90…` | the `truncated_nb`-trained arm, same shape. ⛔ **MEASURED 2026-10-28: produced by `amber` (`2a5efc2a`), and byte-identical in `mu`/`theta`/`gate` to `d4_result_s42.npz` at exactly `0.000000e+00`** — because amber's 190 weight tensors are bit-identical to `silver_0_1_4`'s (M74). ⇒ a redundant copy, the same duplication this table already flags two rows above; and ⛔ it must never be counted as a fourth replicate. |
| `d4_result_s42.npz` | 135 MB | `7f5dd77faf175d5b…` | ⭐ RP-2 seed replicate, **seed 42**, vehicle `c7745155`, family `truncated_nb` @ dropout 0.15, git `2e73de3`, 300 lessons. One of the **three** that give the B-1a/B-1b noise floor (M75). ⛔ Never pooled with `d4_result_tnb.npz` — see below. |
| `d4_result_s43.npz` | 136 MB | `34db0cb22a0aca16…` | ⭐ RP-2 seed replicate, **seed 43**, vehicle `80696bb5`, family `truncated_nb` @ dropout 0.15, git `2e73de3`, 300 lessons. One of the **three** that give the B-1a/B-1b noise floor (M75). ⛔ Never pooled with `d4_result_tnb.npz` — see below. |
| `d4_result_s44.npz` | 137 MB | `7f5e659973502981…` | ⭐ RP-2 seed replicate, **seed 44**, vehicle `23fb3221`, family `truncated_nb` @ dropout 0.15, git `2e73de3`, 300 lessons. One of the **three** that give the B-1a/B-1b noise floor (M75). ⛔ Never pooled with `d4_result_tnb.npz` — see below. |
| `d4_result_tnbNODROP.npz` | 18 MB | `dc175fd5f39fe252…` | the dropout-OFF condition (1 pass × 16), C-128's cost test |
| `p2_result.npz` | 15 MB | `2186c479b8e92dc6…` | the gate-calibration readout (P2) |

**430 MB total, 297 MB after de-duplication.**

## ✅ Archive

**Google Drive**, uploaded 2026-10-22 with `rclone`:

```
gdrive:views-hydranet-artifacts/2026-10-12_underfire_diagnosis/
```

**12 objects, 297.18 MiB.** Every one verified **identical by checksum** after upload
(`rclone check --one-way --combined`, all `=`). To re-verify or restore:

```bash
rclone check  reports/2026-10-12_underfire_diagnosis \
              gdrive:views-hydranet-artifacts/2026-10-12_underfire_diagnosis \
              --filter "+ *.npz" --filter "+ ARTIFACT_MANIFEST.md" --filter "- *" --one-way
rclone copy   gdrive:views-hydranet-artifacts/2026-10-12_underfire_diagnosis <dest>
```

⛔ `d4_result.npz` is **deliberately absent from the archive.** It is byte-identical to
`d4_result_CONTROL_nb_4acbf4d3.npz` (same sha256 locally, same sha1 on Drive), so archiving it
would have cost 133 MB to store the same bytes twice. 430 MB on disk became 297 MB archived.

⚠️ **The archive has an expiry date on it.** `rclone` here uses its own shared Google client ID,
which Google is retiring — it **stops working during 2026**:
<https://rclone.org/drive/#making-your-own-client-id>. A restore attempted after that fails until
someone creates a private client ID. ⇒ Verify the archive before relying on it, not after.

⚠️ **One copy is not a backup.** Drive plus this machine is two copies of 297 MB of irreplaceable
measurement, on one account and one disk. Issue #448 asks the durable-storage question properly.

## Why this needed saying at all

The system disk was at **99% (18 GB free of 938 GB)** when this manifest was first written, there
was no external media and no network volume mounted, and an OOM crash froze the machine hours later.
⛔ All five cubes survived that crash — **verified in seconds, by checksum, because this file
existed.** That is the entire argument for writing it.

**What a reader should do before trusting a number from this dossier:** check that the file is still
present and its sha256 still matches. If it does not, the number is unverifiable and should be
re-measured or withdrawn, not quoted.
