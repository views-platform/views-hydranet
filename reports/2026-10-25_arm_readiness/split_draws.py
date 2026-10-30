"""Split a d4 cube along its MC-dropout draw axis, so B-1's draw-count bias can be swept.

⛔ WHY THIS IS A COMMITTED INSTRUMENT AND NOT A SCRATCHPAD SCRIPT.  The control's splits
(`d4_result_hf0125_s0..s3`, `_p1A..p3B`) were made ad hoc on 2026-10-25 and the code died with
the shell.  §4b of the arm pre-registration then SCORED P-4 on the 1/n intercept those splits
produced — i.e. the pre-registered estimator for the headline prediction existed only as a
command someone had typed.  That is C-106's shape (a gate on no execution path) applied to an
estimator, and it is why the arm's splits are produced by this file instead.

WHAT IT WRITES, for a cube with n draws (here n = 4):
  <tag>_s0 .. _s{n-1}   the n single-draw cubes        -> the n = 1 point
  <tag>_p1A/B, p2A/B, p3A/B   the 3 DISJOINT 2-vs-2 partitions -> the n = 2 point and,
                              A against B, the split-half noise floor at equal draw count

⛔ THE PARTITIONS ARE THE THREE DISJOINT PAIRINGS OF FOUR DRAWS, not three random draws:
{01|23}, {02|13}, {03|12}.  A random pairing could repeat a split and the three |A-B| values
would not be independent reads of the floor.

⛔ EVERY NON-DRAW KEY IS COPIED BYTE-FOR-BYTE, `sha` and `family` included.  A split that
silently dropped `sha` would produce cubes whose vehicle cannot be identified, and
`_family_of` in the readouts raises on a missing `family` rather than assuming `nb` — so a
dropped key fails loud there instead of producing a plausible table here.

usage:  split_draws.py --tag=_ARM_hf0126
"""

from __future__ import annotations

import resource
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).parent.parent / "2026-10-12_underfire_diagnosis"
DRAW_KEYS = ("mu", "theta", "gate")

TAG = ""
for _a in sys.argv[1:]:
    if _a.startswith("--tag="):
        TAG = _a.split("=", 1)[1]
if not TAG:
    raise SystemExit("usage: split_draws.py --tag=_ARM_hf0126   (an untagged write would "
                     "overwrite the control's own cube — see d4_mu_on_the_grid.py's TAG note)")

src = SP / f"d4_result{TAG}.npz"
if not src.exists():
    raise SystemExit(f"no such cube: {src}")
z = np.load(src)

n = z[DRAW_KEYS[0]].shape[0]
if n != 4:
    raise SystemExit(
        f"VOID: this cube has {n} draws, not 4. The three disjoint 2-vs-2 partitions below are "
        "the pairings of FOUR draws; with any other n the n=2 point and the noise floor would "
        "be computed over a different design than the control's and the two would not compare."
    )
carried = {k: z[k] for k in z.files if k not in DRAW_KEYS}
print(f"source {src.name}  draws {n}  carried keys {sorted(carried)}")


def write(suffix: str, idx: tuple[int, ...]) -> None:
    out = SP / f"d4_result{TAG}_{suffix}.npz"
    np.savez_compressed(out, **{k: z[k][list(idx)] for k in DRAW_KEYS}, **carried)
    print(f"  {out.name:44s} draws {idx}  {out.stat().st_size / 1e6:.1f} MB")


for i in range(n):
    write(f"s{i}", (i,))

# The 3 disjoint pairings: draw 0 pairs with each of 1, 2, 3 in turn, and its
# complement is forced. Enumerating that way makes the disjointness structural, not checked.
for p, j in enumerate(range(1, n), start=1):
    a = (0, j)
    b = tuple(k for k in range(n) if k not in a)
    assert not set(a) & set(b) and len(a) + len(b) == n, "partition is not a partition"
    write(f"p{p}A", a)
    write(f"p{p}B", b)

print("\nCOST, measured")
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2))
