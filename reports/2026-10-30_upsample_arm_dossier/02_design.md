# 02 — Design

**2026-10-30.** ⭐ Graduates to a proposed ADR **only on `promote`**, i.e. only if the arm wins.

## 1 · The defect, derived and measured

`views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py:216` and `:224`:

```python
nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2, padding=0, output_padding=0)
nn.ConvTranspose2d(base * 2, base,     2, stride=2, padding=0, output_padding=0)
```

**`kernel_size == stride == 2` ⇒ the kernel windows do not overlap**, so

```
out[2i+a, 2j+b] = W[:, :, a, b] @ x[i, j]        a, b ∈ {0, 1}
```

Each coarse cell becomes a 2×2 block of **four fixed per-channel multiples of the same scalar**.
⇒ **No spatial mixing occurs. The stage replicates and rescales; it never interpolates**, and the
four phases are free to diverge.

**Measured** (`M92`, `up1`, dropout off, teacher-forced): ρ(1) **+0.123146** against ρ(2)
**+0.920900**, channel-wise ρ(1) spanning **−0.724900 … +0.718700** — negative meaning adjacent
cells *anti*-correlated. **Untrained** (`G-F`) the signature is **stronger**: gap **+1.257220**,
floor **−0.903100**, ρ(1) itself **−0.299100**.

**End to end** (`M92` + `M93`): fed ρ(1) **+0.344210** → encoder **+0.313470** / **+0.443300**
(**roughness-neutral**, bracketing its own input) → output **+0.958400**.
⇒ **The whole move belongs to the decoder.**

## 2 · The remedy, and why the obvious one is disqualified

`base = total_hidden_channels = 32`. The twelve upsample modules hold **246,336** of the
incumbent's **905,289** parameters.

| remedy | upsample params | total | Δ | verdict |
|---|--:|--:|--:|---|
| `nearest + Conv2d(3×3)` — the textbook fix | 553,536 | 1,212,489 | **+33.9%** | ⛔ **DISQUALIFIED** |
| ⭐ **`nearest + Conv2d(2×2)` = `UpsampleConv2`** | 246,336 | **905,289** | **±0 exactly** | **THE ARM** |
| `nearest + Conv2d(1×1)` = `UpsampleConv1` | 61,440 | 720,969 | −20.4% | registered; follow-up |

⛔ **+33.9% is 5.6× the most capacity-confounded arm in EXP-01** (WideMemory, +6.0%), whose own
table concludes *"no arm is capacity-confounded in the way first feared."* A win **or** a loss would
be uninterpretable. `NearestConvUp` **refuses `kernel_size=3`** with an error naming this section,
so it is not reachable by a later edit.

**What each remedy does to the mechanism:**

- **`1×1`** — every phase applies the **same** matrix to the same replicated value ⇒ **no phase
  pattern can exist, by construction.** No mixing in the stage; mixing stays with
  `dec_conv0`/`dec_conv1`, which are already 3×3 and already do it.
- **`2×2`** — at `(even, even)` the window sees four copies of one input cell and applies the
  **summed** kernel; at the other three phases it **straddles block boundaries and mixes
  neighbours.** ⇒ Residual phase structure is a **constraint on shared weights**, not a free
  parameter — at **zero** parameter cost.

⚠️ **`nearest`, not `bilinear`:** bilinear is itself a smoother and would push in the **same
direction as the defect**, confounding the arm with the thing it removes.

## 3 · The arm was chosen by measurement, not by argument

`PREREG.md` §2 pre-committed: *the remedy that removes the signature at the lowest parameter cost
takes the GPU slot.* `G-F` measured all three untrained on one seed:

| | gap `ρ(2)−ρ(1)` at `up1` | channel floor | params |
|---|--:|--:|--:|
| control | **+1.2572** | **−0.9031** | 905,289 |
| `UpsampleConv2` | **−0.1984** | **+0.6175** | **905,289** |
| `UpsampleConv1` | **−0.2078** | **+0.6047** | 720,969 |

⇒ Both remove it; **`UpsampleConv2` removes it for free.** ⚠️ The 0.0094 separation between the two
is **not** a basis for preferring either — one seed, untrained, no σ̂ — and `P-21` is logged as a
weak hold for exactly that reason.

## 4 · Implementation shape, and why inheritance is right here

`views_hydranet/architectures/upsample_interp.py`: a `NearestConvUp(nn.Module)` replacement stage,
a `_UpsampleInterpBase(HydraBNUNet06_LSTM4)` that replaces **all twelve**
`upsample{0,1}_head{1,2,3}_{reg,class}` modules in `__init__`, and two concrete subclasses
differing only in `_KERNEL`.

- ⭐ **Widths are read off the module being replaced**, never recomputed. Restating the parent's
  channel plan would duplicate it and the copy would diverge silently.
- ⭐ **A fail-loud guard** refuses to build if a stage is not a `ConvTranspose2d` with
  `kernel_size == stride` — otherwise a future parent change would yield an arm that tests nothing
  while looking healthy. **Mutation-checked:** deleting the guard fails its test.
- ⛔ **All twelve, not just `reg`.** EXP-01's finding is that the effect **is** the gate's placement
  (*"the body is untouched; this is entirely the gate's placement degrading through the loop"*), so
  replacing only the body path would make the comparison meaningless.
- `forward` is **inherited verbatim**; shapes are drop-in, so the `torch.cat` after each upsample is
  untouched.
- **Inheritance over composition, deliberately:** all six prior candidates are subclasses that
  override modules in `__init__`. Composition would mean restructuring the parent — out of scope,
  and it would make candidate 7 inconsistent with its siblings.
- **Registered through the `ARCHITECTURE_REGISTRY` extension seam**; the dispatcher is not modified.

## 5 · What is deliberately NOT changed

Training loop, loss, `output_distribution: nb`, `forecast_composition`, `body_supervision`, the
curriculum, `ss_epsilon_max: 0.0`, the queryset, the partition, the scoring protocol, `pos_weight`.
⛔ **The only config key that differs between the arm and its control is `model`.**
