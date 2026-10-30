# G-F — both remedies remove the phase mechanism, and the prereg selects the free one

**2026-10-30.** Script `gf_phase_signature.py` (committed `before` it ran, D-5) + amendment 1
(`C-347`). Output `gf_phase_signature.out`. **283 s CPU, zero GPU.** Pre-registration:
`PREREG.md` §4/§5b/§6.

---

## ⭐⭐⭐ THE ANSWER IN THREE LINES

**1 · The mechanism is real and the instrument detects it.** The **untrained** control reproduces
`M92`'s phase signature **more strongly than the trained checkpoint did**: gap `ρ(2) − ρ(1)` at
`up1` = **+1.2572** (trained: +0.7978), channel floor **−0.9031** (trained: −0.7249).
⇒ **Not void** under `PREREG.md` §6 device 4, whose bar was ≥ 0.40.

**2 · ⭐⭐ Both remedies remove it completely.** The gap goes **negative** — back to the normal
monotone decay where ρ(1) > ρ(2) — and the channel floor goes **positive**, so **no channel is
anti-correlated any more.**

**3 · ⛔ But `out`'s smoothness barely moves, and that is the honest limit.** ρ(1) at
`dec_conv4_head1_reg` is **+0.5822** (control) → **+0.5702** / **+0.5559**. ⇒ **`G-F` shows the
mechanism is gone. It does NOT show the output field becomes rougher.** §4.

---

## 1 · The measurement

| mode | stage | ρ(1) | ρ(2) | **GAP** | ch floor ρ(1) |
|---|---|--:|--:|--:|--:|
| **control** `HydraBNUNet06_LSTM4` | `up0` | −0.0003 | +0.7063 | **+0.7066** | −0.5743 |
| **control** | **`up1`** | **−0.2991** | **+0.9582** | ⛔ **+1.2572** | ⛔ **−0.9031** |
| **control** | `out` | +0.5822 | +0.4476 | −0.1346 | +0.4723 |
| `UpsampleConv2` **(+0.0%)** | `up0` | +0.7820 | +0.5862 | −0.1958 | +0.4692 |
| `UpsampleConv2` | **`up1`** | **+0.7920** | +0.5936 | ⭐ **−0.1984** | ⭐ **+0.6175** |
| `UpsampleConv2` | `out` | +0.5702 | +0.4231 | −0.1471 | +0.3624 |
| `UpsampleConv1` **(−20.4%)** | `up0` | +0.7740 | +0.5489 | −0.2251 | +0.5749 |
| `UpsampleConv1` | **`up1`** | **+0.7912** | +0.5833 | ⭐ **−0.2078** | ⭐ **+0.6047** |
| `UpsampleConv1` | `out` | +0.5559 | +0.4222 | −0.1337 | +0.3736 |

⭐ **Read `up1`'s ρ(1) column**: the control is **−0.2991** — adjacent cells *anti*-correlated on
the median channel, which is the phase pattern in its purest form. Both remedies sit at **+0.79**,
the ordinary near-field correlation of a smooth field.

**Gates:** `G-17` ρ controls (ramp **+0.9996**, checkerboard **−1.0000**, iid noise **+0.0006**);
`G-23` parameter counts **905,289 / 905,289 / 720,969**, exactly `PREREG.md` §2; `G-24` no
`_BatchNorm` in train mode, 15 `LockedDropout` in eval, for every mode (`C-384`); `G-25` two
forwards bit-identical, every mode; `G-26` see §3.

---

## 2 · The pre-registered verdict

| id | declared before the run | measured | verdict |
|---|---|---|---|
| **P-20** | at least one remedy: gap **< 0.15** and channel floor **≥ −0.05** | `UpsampleConv2` −0.1984 / +0.6175; `UpsampleConv1` −0.2078 / +0.6047 | ✅ **HELD, 2 of 2** |
| **P-21** | `UpsampleConv1`'s gap **below** `UpsampleConv2`'s, because a 1×1 kernel cannot form a phase pattern at all while a 2×2 can | −0.207828 vs −0.198421 | ⚠️ **HELD, but on a margin of 0.009 — see below** |

⚠️ **P-21 is a weak hold and I am not claiming it as a win.** Both gaps are **negative**, so the
signed comparison the prediction asked for is satisfied — but the two differ by **0.0094** on a
one-seed, untrained, no-σ̂ quantity. ⇒ **The honest statement is that both remedies remove the
mechanism and this measurement cannot separate them.** The structural argument (1×1 cannot form a
phase pattern; 2×2 can) is **untested by this run** and would need trained weights.

## ⭐⭐ ⇒ The arm is `UpsampleConv2`, selected by the committed rule

`PREREG.md` §2: *"the remedy that removes the phase signature at the **lowest parameter cost** takes
the GPU slot."*

| remedy | removes it? | params | Δ |
|---|---|--:|--:|
| ⭐⭐ **`UpsampleConv2`** | ✅ | **905,289** | **+0.0% — exact** |
| `UpsampleConv1` | ✅ | 720,969 | −20.4% |

⇒ **`UpsampleConv2` at ±0 parameters.** This is the `AntiAliasedPool` standard, which the
2026-08-24 dossier calls *"the cleanest possible experiment in the whole space"* **because** a
zero-parameter remedy means **a win cannot be attributed to capacity.**

⚠️ `UpsampleConv1` removed marginally more and costs capacity in the other direction. ⛔ **The rule
stands as written** — changing it now would be changing a prereg after seeing its numbers, which
`PREREG.md` §6 device 6 forbids. `UpsampleConv1` stays registered and available as a follow-up.

---

## 3 · ⛔⛔ A defect this run found in how every measurement this week was launched (`C-347`)

The **first** run of `G-F` died with `Unknown model type: 'UpsampleConv2'` — because it had imported
`views_hydranet` from **site-packages (installed 0.1.5)**, not the working tree.

**The mechanism:** for a script, `sys.path[0]` is **the script's own directory**, not the cwd. So
`python reports/.../x.py` never sees the repo. ⭐ **`pytest` passes regardless**, because it puts the
repo root on `sys.path` ⇒ **the test suite and the measurement scripts were reading different
code.**

⛔ **It failed loudly only because the name did not exist in 0.1.5.** Had it existed, `G-F` would
have measured a different build and printed clean numbers.

⭐ **Exposure checked rather than assumed.** Installed 0.1.5 is **byte-identical** to the working
tree for every module this week's measurements executed — `HydraBNrecurrentUnet_06_LSTM4.py`,
`locked_dropout.py`, `nb_core.py`, `composition.py`, `utils/utils.py`. Only `registry.py` differs,
and only by the two lines added today. ⇒ ✅ **`M90`, `M91`, `M92` and `M93` are UNAFFECTED.**

**Fixed by amendment 1:** `REPO` goes on `sys.path` first, and **`G-26` asserts the resolved
`views_hydranet.__file__` is under `REPO` and prints it.** A comment is not a guard.

⚠️ **Not fixed here, and it should be:** no other script in `reports/` carries `G-26`. This is a new
instance of `C-347` with the arrow pointed at `reports/`, and it belongs in the risk register —
⛔ which has hand-maintained header counts and its own skill, so it is **flagged for Simon, not
edited.**

---

## 4 · ⚠️ LIMITS — and the third one is the one that matters

- ⛔⛔ **`out`'s smoothness barely moved: +0.5822 → +0.5702 / +0.5559.** ⇒ **`G-F` establishes that
  the phase mechanism is gone, NOT that the delivered field becomes rougher.** ⚠️ And note the
  scale: the **trained** control's `out` ρ(1) is **+0.958** against **+0.582** untrained, so
  **output smoothness is overwhelmingly a TRAINED property** that this run cannot speak to.
  **That is precisely what the arm tests.**
- ⛔ **UNTRAINED weights.** This measures the stage's **geometry**, not what training does with it.
  **A trained `UpsampleConv2` could re-learn a phase pattern its 2×2 kernel permits** — the 1×1
  cannot, and that asymmetry is a real reason to keep `UpsampleConv1` live.
- ⛔ **TEACHER-FORCED** (`C-399`), to match `M92`. The arm is scored **autoregressively**, and
  EXP-01's entire finding is that **the two do not agree.**
- ⚠️ **One seed, no σ̂.** The architecture difference is deterministic given the seed; the
  **magnitude** of each gap is a one-draw quantity.
- ⚠️ `sb` head only, `reg` branch only, three stages hooked.
- ⛔ **Per `S-2`, nothing here closes anything, and `G-F` was never a gate on whether the arm
  runs.** The arm's premise is `M92`, already measured. `G-F` only picked the remedy.
