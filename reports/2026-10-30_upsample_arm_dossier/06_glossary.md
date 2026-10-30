# 06 — Glossary

**2026-10-30.** ⛔ Terms this programme introduces, defined once. The repository's locked vocabulary
is `reports/GLOSSARY.md` — ⛔ **edit that file, never invent a synonym here.** These are additions
pending inclusion there.

| term | definition |
|---|---|
| **phase** | one of the four positions `(a, b) ∈ {0,1}²` inside the 2×2 output block that a stride-2 upsample produces from a single input cell |
| **phase pattern** / **phase striping** | a systematic difference between the four phases. In the incumbent each phase has its **own free weight matrix**, so adjacent output cells are fixed multiples of the same scalar and can be **anti**-correlated. ⛔ Distinct from blurring: it **lowers** ρ(1) while leaving ρ(2) high |
| **the gap** | `ρ(2) − ρ(1)` at a stage, over land–land neighbour pairs. The programme's signature statistic: a phase pattern makes it **large and positive**; ordinary spatial decay makes it **negative** |
| **channel floor** | the **minimum** across channels of ρ(1) at a stage. Reported beside the median because a negative floor is the direct fingerprint of anti-correlated neighbours |
| **`NearestConvUp`** | the replacement stage: `nn.Upsample(scale_factor=2, mode="nearest")` followed by `nn.Conv2d(kernel_size ∈ {1, 2})`. Output shape is identical to the `ConvTranspose2d(k=2, s=2)` it replaces |
| **`UpsampleConv2`** | candidate 7a — `NearestConvUp(kernel_size=2)` on all twelve upsample stages. **±0 parameters.** ⭐ The arm |
| **`UpsampleConv1`** | candidate 7b — `NearestConvUp(kernel_size=1)`. **−20.4% parameters**; no phase pattern can form by construction. Registered, measured by `G-F`, **not queued** |
| **`G-F`** | the CPU mechanism probe: three modes **untrained** on one seed, which selected the arm by the pre-committed lowest-parameter-cost rule |
| **roughness-neutral** | a stage whose output ρ(1) sits **between** or at its input's — i.e. it neither smooths nor sharpens. Measured of the encoder (+0.313 / +0.443 against a fed field at **+0.344**) |

⚠️ **Retired on arrival: "regime".** It was used in `M90` §4 for *bins of the model's own predicted
spread* and read, reasonably, as *latent states of the world*. ⇒ Use **stratum** or **bin** for a
partition of a predicted statistic, and reserve **state** for a latent-variable model.
