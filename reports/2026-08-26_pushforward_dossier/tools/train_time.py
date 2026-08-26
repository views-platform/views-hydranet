#!/usr/bin/env python3
"""Extract TRAINING wall-clock from a smoke log — not total runtime.

The smoke's total is dominated by emit + scoring, which the pushforward does not touch: on the
2-lesson control, training is ~45 s of a ~1400 s run. A ratio of totals would therefore report a
cost multiplier near 1.0 and understate a 300-lesson arm by more than an order of magnitude, because
at 300 lessons training is nearly all of the time (300/2 x 45 s ~ 1.9 h, matching the incumbent's
measured 1.82 h/arm).

Reads the tqdm summary line the training loop leaves behind: `2082/2082 [MM:SS<00:00, ...]`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: the COMPLETED tqdm bar only — `<00:00` means zero remaining, so a mid-run line cannot match.
_DONE = re.compile(r"(\d+)/(\1) \[(\d\d):(\d\d)<00:00,")


def training_seconds(log: Path) -> int:
    """Seconds the training loop took, or raise if the log has no completed bar."""
    text = log.read_text(errors="replace")
    hits = _DONE.findall(text)
    if not hits:
        raise SystemExit(
            f"{log.name}: no completed training bar found. The run did not finish training, or "
            "tqdm's format changed — do not guess a number from the total runtime."
        )
    # last bar wins: BN recalibration can emit its own
    _, _, mm, ss = hits[-1]
    return int(mm) * 60 + int(ss)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        p = Path(arg)
        print(f"{p.name}: training {training_seconds(p)} s")
