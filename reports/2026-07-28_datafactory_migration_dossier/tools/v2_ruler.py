"""v2 ruler adapter (Epic #203 S4) — frozen datafactory TRUTH + a thin wrapper over the
byte-identical lodestar scorer.

The clean-cut re-anchors only the TRUTH the ruler scores against (v1 viewser → v2 datafactory). The
scoring *functions* (`crps_ensemble`, AP/Brier, `score_models`) are reused byte-identical from the
frozen v1 lodestar dossier — this module does NOT reimplement them, it points them at v2 truth.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
V2_TRUTH = _HERE / "v2_truth" / "calibration_datafactory_df.parquet"
V2_TRUTH_SHA256 = "620f4aa3722fec9fb3965ada5ed3f041ae45a72a5937b5ac748453847d2ce84b"
# The frozen v1 scorer (byte-identical functions; only the truth arg changes).
_LODESTAR = (
    _HERE.parent.parent / "2026-07-17_lodestar_eval_dossier" / "tools" / "lodestar_score.py"
)


def load_lodestar():
    """Import the frozen lodestar scorer module (byte-identical scoring functions)."""
    spec = importlib.util.spec_from_file_location("lodestar_score", str(_LODESTAR))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def v2_support():
    """Return the v2 truth support (months, cells) — the identical-support discipline anchor."""
    df = pd.read_parquet(V2_TRUTH)
    mi = df.index.get_level_values("month_id")
    pg = df.index.get_level_values("priogrid_id")
    return {
        "months": (int(mi.min()), int(mi.max()), int(mi.nunique())),
        "cells": int(pg.nunique()),
    }


def score_v2(registry, targets=("sb", "ns", "os")):
    """Score model prediction dirs against the v2 (datafactory) truth via the frozen lodestar
    `score_models`. `registry` is the lodestar registry list of
    "<label>|<pred_dir>|lr_{t}_best|by_{t}_best" specs."""
    if not V2_TRUTH.exists():
        raise FileNotFoundError(f"v2 truth missing: {V2_TRUTH} — freeze it (Epic #203 S4)")
    return load_lodestar().score_models(str(V2_TRUTH), registry, list(targets))
