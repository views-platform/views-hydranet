"""mask_pairing_gate.py — G-3 for RP-2: do the three replicates see the SAME dropout masks?

⛔ WHY THIS GATES THE 37 MINUTES. `d4_mu_on_the_grid.py` calls `lock_entropy(42, 42)` before
`load_vehicle`, and `LockedDropout` draws its mask from the GLOBAL torch RNG, so all three runs
*should* see identical masks -- which makes the sigma-hat design PAIRED and cancels the
draw-count bias as a common offset. But `lock_entropy` sets
`use_deterministic_algorithms(..., warn_only=True)`, so non-determinism can fall through.
⛔ If the masks differ, sigma-hat contains draw noise and B-1a moves 0.755 -> 0.529 -> 0.400
over n = 1, 2, 4 ⇒ draw noise wearing a seed-effect costume. Hard stop and re-plan.

⭐ NARROWED, not dropped, by M74. M74 measured TRAINING determinism as exact (amber == silver ==
silver_0_1_4, 190/190 bit-identical across two git shas and three library versions). That is
stronger evidence than this gate was designed to produce -- but it is evidence about TRAINING.
This gate is about the SCORING forward pass, which is a different code path, so it still runs.

**Why a synthetic input is sufficient, and not a shortcut.** `LockedDropout.forward` keys its cache
on `(tuple(x.shape), x.device, x.dtype)` and draws `torch.rand_like(x)`. ⇒ the mask VALUES
depend on the global RNG state and the input SHAPE, not at all on its values. A zeros volume
of the real shape therefore draws the identical mask sequence a real volume would. The
shape is taken from `build_volume` on one month rather than hard-coded, so it cannot
drift from d4's.

C-351 slots: population = the 3 replicate checkpoints (+2 control runs) · transform = raw mask
tensors as drawn · estimator = sha256 over mask bytes, plus exact `torch.equal` · not normalised.

================================================================================================
GATES, DECLARED BEFORE THE RUN
================================================================================================
G-3a ⭐ THE PAIRING GATE. All three replicates must produce the SAME mask keys AND byte-identical
     mask tensors under d4's own preamble. ⇒ the design is paired.

G-3b ⛔ NON-VACUITY, and without it G-3a is worthless. A fourth load under a DIFFERENT entropy lock
     (7, 7) must produce DIFFERENT masks. If the masks are identical even then, this script is
     reporting a constant -- not a pairing -- and G-3a carries no information.

G-3c ⛔ THE MASKS MUST ACTUALLY DROP. At least one mask must contain a zero, and the kept-fraction
     must sit near 1 - p. An all-ones mask would make G-3a pass for the trivial reason that
     dropout is inert (the C-324 inert-knob signature).

G-3d The three replicates must be the three SHAs the allowlist admits, read from the loader,
     and amber must NOT be among them (different pod session, git 73c3f42; views-models
     forbids pooling it).

MY RISKY PREDICTION, before execution: G-3a PASSES. M74 showed the RNG lock survives a full
training run bit-exactly across library versions, so a single forward after the same lock is very
unlikely to diverge. ⚠️ The honest counter-case: `load_vehicle` consumes RNG draws during model
CONSTRUCTION, so if the three configs caused a different number of draws the states would
diverge -- the configs differ only in the two seed VALUES, which are not construction
parameters, so this should not happen. If G-3a fails, that is the mechanism to look at first.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path("reports/2026-10-12_underfire_diagnosis").resolve()))

from vehicle import build_volume, load_vehicle, pull  # noqa: E402

from views_hydranet.architectures.locked_dropout import LockedDropout  # noqa: E402
from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate  # noqa: E402

BACKUP = pathlib.Path("/home/simon/Documents/kestrel_sigma_backup_20261021")
#: ⛔ The allowlist. amber is REFUSED BY NAME (different pod session, git 73c3f42).
REPLICATES = {
    "silver_s42": "silver_0_1_4/calibration_model_20261021_092504.pt",
    "bronze_s43": "bronze/calibration_model_20261021_010229.pt",
    "cobalt_s44": "cobalt/calibration_model_20261021_005527.pt",
}
SP = pathlib.Path("reports/2026-10-12_underfire_diagnosis")

torch.set_num_threads(8)
fails: list[str] = []


def gate(ok: bool, name: str, detail: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


print("=" * 96)
print("G-3 — the mask-pairing gate, on the SCORING forward pass")
print("=" * 96)

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
vol, _, _ = build_volume(df, [421])
print(f"  volume shape from build_volume (NOT hard-coded): {tuple(vol.shape)}")


def masks_for(rel: str, np_seed: int = 42, torch_seed: int = 42) -> tuple[dict, str, int]:
    """d4's own preamble, then ONE forward, then the masks."""
    ReproducibilityGate.lock_entropy(np_seed=np_seed, torch_seed=torch_seed)
    model, cfg, sha = load_vehicle(str(BACKUP / rel))
    model.set_locked_dropout(True)
    model.reset_locked_dropout()
    n_do = sum(1 for m in model.modules() if isinstance(m, LockedDropout))
    with torch.no_grad():
        h = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:])
        model(vol[:, 0], h)
    out: dict[tuple, torch.Tensor] = {}
    for i, m in enumerate(model.modules()):
        if isinstance(m, LockedDropout):
            for k, t in m._masks.items():
                out[(i, k[0])] = t.clone()
    return out, sha, n_do


def digest(ms: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(ms, key=str):
        h.update(str(k).encode())
        h.update(ms[k].numpy().tobytes())
    return h.hexdigest()[:16]


print("\n  loading each replicate under lock_entropy(42, 42) and taking ONE forward")
got: dict[str, tuple[dict, str, int]] = {}
for name, rel in REPLICATES.items():
    got[name] = masks_for(rel)
    ms, sha, n_do = got[name]
    print(
        f"    {name:<12} sha {sha[:8]}  {n_do} LockedDropout modules  "
        f"{len(ms)} cached masks  digest {digest(ms)}"
    )

print("\n" + "=" * 96)
print("  GATES")
print("=" * 96)

names = list(REPLICATES)
digests = {n: digest(got[n][0]) for n in names}
keys_same = len({tuple(sorted(got[n][0], key=str)) for n in names}) == 1
exact = all(
    torch.equal(got[names[0]][0][k], got[n][0][k]) for n in names[1:] for k in got[names[0]][0]
)
gate(
    keys_same and exact and len(set(digests.values())) == 1,
    "G-3a",
    f"all three replicates share mask keys ({keys_same}) and byte-identical tensors ({exact}); "
    f"digests {sorted(set(digests.values()))}",
)

alt, _, _ = masks_for(REPLICATES["silver_s42"], np_seed=7, torch_seed=7)
gate(
    digest(alt) != digests["silver_s42"],
    "G-3b",
    f"NON-VACUITY: lock_entropy(7,7) gives digest {digest(alt)} vs "
    f"{digests['silver_s42']} at (42,42) — the hash is not a constant",
)

ms0 = got[names[0]][0]
has_zero = any(bool((t == 0).any()) for t in ms0.values())
kept = sum(float((t > 0).sum()) for t in ms0.values()) / sum(t.numel() for t in ms0.values())
gate(
    has_zero and 0.80 < kept < 0.92,
    "G-3c",
    f"the masks actually drop: a zero is present ({has_zero}), kept fraction {kept:.4f} "
    f"(expect ~0.85 at p=0.15)",
)

shas = {n: got[n][1] for n in names}
want = {"c7745155", "80696bb5", "23fb3221"}
gate(
    {s[:8] for s in shas.values()} == want and "2a5efc2a" not in {s[:8] for s in shas.values()},
    "G-3d",
    f"allowlist: loaded {sorted(s[:8] for s in shas.values())}, amber(2a5efc2a) absent",
)

print("\n" + "=" * 96)
if fails:
    print(f"  ⛔ HARD STOP — failed: {', '.join(fails)}")
    print("     The sigma-hat design is NOT paired. Do not spend the 37 minutes; re-plan.")
else:
    print("  ALL GATES PASS — the design is paired; the three d4 runs may proceed.")
print("=" * 96)
sys.exit(1 if fails else 0)
