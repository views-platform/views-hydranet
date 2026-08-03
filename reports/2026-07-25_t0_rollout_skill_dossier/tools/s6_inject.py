#!/usr/bin/env python3
"""S6 (#199) eval-config injector for the bloom-verification matrix.

Produce an eval config for one arm from a canonical base (cfg_nb_sample.py):
  set output_distribution, forecast_composition, rollout_feedback, torch/np seed,
  and gate_threshold ONLY for threshold_gate (the validator rejects it otherwise).

Usage: s6_inject.py <base> <dst> <od> <comp> <fb> <seed>
"""
from __future__ import annotations

import re
import sys


def inject(base: str, dst: str, od: str, comp: str, fb: str, seed: int) -> str:
    t = open(base).read()

    # 1) output_distribution value
    t, n = re.subn(
        r"('output_distribution'\s*:\s*)'[a-z_]+'", r"\1'%s'" % od, t, count=1
    )
    assert n == 1, "output_distribution not replaced"

    # 2) rollout_feedback value (base has it; replace)
    t, n = re.subn(r"('rollout_feedback'\s*:\s*)'[a-z_]+'", r"\1'%s'" % fb, t, count=1)
    assert n == 1, "rollout_feedback not replaced"

    # 3) torch_seed / np_seed values
    t, n = re.subn(r"('torch_seed'\s*:\s*)\d+", r"\g<1>%d" % seed, t, count=1)
    assert n == 1, "torch_seed not replaced"
    t, n = re.subn(r"('np_seed'\s*:\s*)\d+", r"\g<1>%d" % seed, t, count=1)
    assert n == 1, "np_seed not replaced"

    # 4) forecast_composition: replace if present, else insert after rollout_feedback line
    if re.search(r"'forecast_composition'\s*:", t):
        t = re.sub(
            r"('forecast_composition'\s*:\s*)'[a-z_]+'", r"\1'%s'" % comp, t, count=1
        )
    else:
        t, n = re.subn(
            r"('rollout_feedback'\s*:\s*'[a-z_]+',)",
            r"\1\n        'forecast_composition': '%s'," % comp,
            t,
            count=1,
        )
        assert n == 1, "could not insert forecast_composition"

    # 5) gate_threshold: present ONLY for threshold_gate (τ=0.5 — validated deployable
    #    th_gated_NB; baserate is a documented no-op per #167 exp-log). Remove otherwise.
    t = re.sub(r"\n\s*'gate_threshold'\s*:\s*[0-9.]+,", "", t)  # strip any existing
    if comp == "threshold_gate":
        t, n = re.subn(
            r"('forecast_composition'\s*:\s*'threshold_gate',)",
            r"\1\n        'gate_threshold': 0.5,",
            t,
            count=1,
        )
        assert n == 1, "could not insert gate_threshold"

    open(dst, "w").write(t)
    return t


if __name__ == "__main__":
    base, dst, od, comp, fb, seed = sys.argv[1:7]
    inject(base, dst, od, comp, fb, int(seed))
    print(f"wrote {dst}: od={od} comp={comp} fb={fb} seed={seed}")
