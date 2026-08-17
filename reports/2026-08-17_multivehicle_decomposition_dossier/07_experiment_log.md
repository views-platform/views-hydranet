# Experiment log — multi-vehicle decomposition

Append-only. Falsifier verdicts recorded **before** predictions are read.

---

### EXP-01 · the decomposition on every vehicle the gate admitted · 2026-08-17/18 · **REPLICATES 4/4**

- **Plan (pre-reg):** `05_analysis_plan.md`, locked before any arm ran.
- **Vehicles:** chosen by `scripts/floor_gate.py` FG-A on data already on disk, not by hand.
  Admitted `violet_visitor` (28.3× chance), `purple_alien` (21.6×), `blue_stranger` (13.9×),
  `blazing_meteor` (9.6×). **Excluded `bright_starship` (2.2×) and `pink_pirate` (0.89× — below random
  ranking).**
- **Arms:** 5 per vehicle, inference-only on the existing artifact, 15 arms in ~4 h. Control is each
  vehicle's preserved production cube, already scored — not re-run.

#### Falsifiers — recorded first

| vehicle | verdict | oracle−control gap @h18 | `thin` ratio (want 0.25 ± 5%) |
|---|---|--:|--:|
| `purple_alien` | **ALL PASS** | 0.2663 | 0.2524 |
| `blue_stranger` | **ALL PASS** | 0.3028 | 0.2507 |
| `blazing_meteor` | **ALL PASS** | 0.3498 | 0.2514 |

F1 (h1 identical across arms), F2 (`N` = 170430 everywhere), F3 (every transform moved its own axis on
that vehicle's real field), F4 (a gap large enough to decompose) — all clear on all three.

#### The result

Share of the oracle→free-running gap recovered, h18, target `sb`:

| vehicle | body | composition | occurrence | magnitude | `thin:0.75` | scrambled |
|---|---|---|--:|--:|--:|--:|
| `violet_visitor` | `nb` | soft_gate | 95.3% | 1.4% | 95.5% | −93.7% |
| `purple_alien` | `mixture_nb` | soft_gate | 89.5% | −1.0% | 85.8% | −48.8% |
| `blue_stranger` | `mixture_nb` | soft_gate | 93.0% | −0.2% | 86.8% | −31.6% |
| `blazing_meteor` | `nb` | **threshold_gate** | 94.3% | **0.0%** | 82.6% | −21.2% |

#### Predictions

| # | verdict | |
|---|---|---|
| **P1** occurrence > magnitude on every eligible vehicle | **HOLDS 4/4** | the claim that matters |
| **P2** occurrence ≥ 70% | **HOLDS 4/4** | range 89.5–95.3% |
| **P3** `spatial_scramble` below the control | **HOLDS 4/4** | all negative |
| **P4** `thin:0.75` ≥ 60% | **HOLDS 4/4** | range 82.6–95.5% |

#### What this establishes

**Occurrence carries 89.5–95.3% of the gap; magnitude carries −1.0% to +1.4% — zero.** Across two body
families (`nb`, `mixture_nb`), two compositions (`soft_gate`, `threshold_gate`), four seeds, four
independent training runs, and a 3× range in baseline retention (0.21 → 0.54).

`blazing_meteor` is the sharpest single number: **0.0%**. Hand that model perfectly correct magnitudes
while keeping its own placement and it recovers *nothing at all*.

**A model can be wildly wrong about magnitude and lose almost nothing.** `purple_alien`'s own magnitudes
are **6.6× off** the real ones (measured: `mean_magnitude_on_active` differs by 664%) and it still
recovers 89.5% of the gap when given correct occurrence. Violet's are 71% inflated with the same result.

**And `thin:0.75` recovers 82.6–95.5% everywhere.** Discard three quarters of the true events, keep the
rest in the right places, and almost nothing is lost.

#### What varies, and should not be quoted as if it did not

`spatial_scramble` is negative on all four — always below the control — but its *magnitude* ranges
−21.2% to −93.7%, a 4× spread. **The sign is the robust part; the size is not.** This is why P1 was
pre-registered on ordering rather than on a number, and it is the same discipline the placement dossier
had to learn when the smoke vehicle's +0.9% turned out to be a floor artifact.

#### The exclusions did the work they were designed to do

`pink_pirate` scores **below random ranking** at h18 and `bright_starship` at 2.2×. On such a vehicle a
degradation arm cannot fall below a control that has already fallen, so `spatial_scramble` would have
read as harmless — exactly as it did on `truncated_smoke` (+0.9% against a true −93.7%). Excluding them
**before** running, on an objective threshold, is the judgement nobody made on 2026-08-14.

#### Scope

One seed per vehicle, one target (`sb`), 13 origins, S=16, h\* = 18. These are different
*configurations*, not different seeds — which is arguably the stronger generalisation test, but it is
not a seed study. `spatial_scramble` inherits C-291's confound (destroying clustering also breaks
alignment with the statics). `blue_stranger` and `blazing_meteor` load only with the uncommitted
views-models#404 fix; that key governs *training* and these are inference-only runs on frozen
artifacts, so it cannot affect the numbers — recorded because the dependency is real.
