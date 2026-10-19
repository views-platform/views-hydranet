"""Shared vehicle loading and volume construction for T0/T2/T3 (views-hydranet#396).

Pre-registered in ../2026-10-12_underfire_diagnosis_prereg.md. Written BEFORE the artifact exists
(the predictions are already committed, so writing the code cannot bias the result).

⛔ F6 (amended): the vehicle is VOID unless the sidecar confirms CONFIG identity with the delivery.
Weight identity is explicitly NOT required — the 2026-09-28 artifacts are gone (views-models#549)
and training here is not bit-reproducible (C-317).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# PRIO-GRID raster. Validated 2026-10-12 against the production raw file's own `row`/`col`
# columns on all 13,110 africa cells: exact match, both axes.
GRID_ROWS, GRID_COLS = 360, 720

# The delivered violet_visitor config (views-models, read-only). F6 compares against this.
EXPECTED_SIDECAR = {
    "model": "HydraBNUNet06_LSTM4",
    "input_channels": 3,
    "total_hidden_channels": 32,
    "output_channels": 1,
    "dropout_rate": 0.15,
    "static_channels": [],
    "static_top_skip": True,
    "output_distribution": "nb",
    "reg_activation": None,
    "n_head_samples": 4,
}
# ⛔ THE SIDECAR DOES NOT RECORD THE REGION OR THE GRID SIZE, and the network is fully
# convolutional, so neither the config nor the state-dict shapes can tell an `africa_me_legacy`
# artifact from a `land` one. Found 2026-10-12 by smoke-testing this guard against the local
# artifacts: all four PASS the config check despite being africa-trained. So config identity is
# necessary and NOT sufficient, and these four are denied by hash.
KNOWN_AFRICA_SHA256 = {
    "909f44c0096ee6b5c675f65539fcc31977bbe3ea6bb52438f2ed7b165237ddcb",  # calibration 20260812
    "a7404fe73f8b3eefe815d1abb97fbab7ba1bf1140faf114f96b41c71fad71c81",  # validation  20260908
    "dcadc80d0e1091477cc446a156af95a0e15c9e57cb1befb5dfb18d0b733da5d7",  # validation  20260916
    "9c8508671db7bb1058b824c2573186998d321907c29edeeafd2958edade28615",  # validation  20260917
}

MIN_LESSONS = 300        # the delivered vehicle's length; 40 is smoke-only (retention programme)

TARGETS = ["ged_sb_best", "ged_ns_best", "ged_os_best"]   # datafactory names, input order
SB = 0                                                     # index of sb among the targets
GATE_THRESHOLD = 0.20                                      # violet_visitor, threshold_gate


def grid_rowcol(pg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PRIO-GRID id -> 1-based (row, col). Validated against the production file."""
    return (pg - 1) // GRID_COLS + 1, (pg - 1) % GRID_COLS + 1


def read_provenance(d: Path) -> dict[str, str] | None:
    """Parse `artifact/PROVENANCE` (views-models fix/549, `pod_run_model.sh:368`) -> key: value.

    Written at collect time from values known there, because `$OUT/REHEARSAL` is written at `:445`
    — after collection — and does not live under `artifact/`, so it neither exists yet nor travels
    with what it describes. (My first suggested fix was to copy it at collect time; that would have
    copied a file that did not exist, which the views-models session caught.)
    """
    f = d / "PROVENANCE"
    if not f.exists():
        return None
    out: dict[str, str] = {}
    for line in f.read_text().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def load_vehicle(artifact: str | Path, device: str = "cpu"):
    """Load the trained model from a .pt plus its REQUIRED .pt.config.json sidecar.

    The sidecar is load-bearing, not metadata: `model_artifact_fetcher.py:107` rebuilds the model
    from it and only then loads the state dict. Without it the fetcher's `else` branch calls
    `.to()` on an OrderedDict and raises AttributeError (views-models#549).
    """
    from views_hydranet.utils.utils import choose_model

    p = Path(artifact)
    sidecar = p.with_suffix(".pt.config.json")
    if not p.exists():
        raise SystemExit(f"VOID: no artifact at {p}. See views-models#550.")
    if not sidecar.exists():
        raise SystemExit(
            f"VOID: no config sidecar beside {p.name}. The weights alone cannot be loaded — "
            "views-models#549. Pull `calibration_model_*.pt*`, not `*.pt`."
        )
    cfg = json.loads(sidecar.read_text())

    # ---- F6, amended: CONFIG identity with the delivery. Weight identity not required. ----
    bad = {k: (cfg.get(k), v) for k, v in EXPECTED_SIDECAR.items() if cfg.get(k) != v}
    if bad:
        raise SystemExit(
            "VOID (F6): the artifact's config does not match the delivery's. "
            + "; ".join(f"{k}: got {g!r} expected {e!r}" for k, (g, e) in bad.items())
        )
    if cfg.get("hurdle_nb_theta") is not None:
        # Amendment 1 §2: a learned theta outside the state dict changes what T3 scores.
        raise SystemExit(
            "VOID (T3 re-spec required): sidecar carries hurdle_nb_theta="
            f"{cfg['hurdle_nb_theta']!r}. For an `nb` vehicle it must be None; a mixture_nb "
            "vehicle makes the sidecar part of T3's object. See prereg Amendment 1 §2."
        )

    sha_file = p.with_suffix(".pt.sha256")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    if sha_file.exists() and sha_file.read_text().strip() != sha:
        raise SystemExit(f"VOID: sha256 mismatch for {p.name}.")

    for marker, why in (
        ("INCOMPLETE", "the config sidecar was missing at collection time"),
        ("REHEARSAL", "it is a --rehearsal artifact, trained at a handful of lessons"),
    ):
        if (p.parent / marker).exists():
            raise SystemExit(f"VOID (F6): {marker} sits beside {p.name} — {why}.")

    prov = read_provenance(p.parent)
    if prov is None:
        raise SystemExit(
            "VOID (F6): no artifact/PROVENANCE beside the artifact. Lessons and region cannot be "
            "established without it, and §4's F6 is VOID rather than reinterpret. Re-collect with "
            "views-models fix/549-collect-pod-artifact (>= f99413ef), which writes it at "
            "pod_run_model.sh:368."
        )
    if "REHEARSAL" in prov.get("mode", ""):
        raise SystemExit(f"VOID (F6): PROVENANCE mode is {prov['mode']!r}.")
    lessons = prov.get("lessons", "unknown")
    if not lessons.isdigit() or int(lessons) < MIN_LESSONS:
        raise SystemExit(
            f"VOID (F6): PROVENANCE lessons is {lessons!r}; the delivered vehicle trained at "
            f"{MIN_LESSONS}. A shorter run is a different model (M45's control was measured at "
            "L=300, and 40 lessons is smoke-only per the retention programme)."
        )

    if sha in KNOWN_AFRICA_SHA256:
        raise SystemExit(
            f"VOID (F6): {p.name} is a known africa_me_legacy artifact (13,110 cells). "
            "The config check cannot catch this — the sidecar records no region and the net is "
            "fully convolutional — so it is denied by hash. Use the artifact from the "
            "views-models#550 pod run."
        )

    model = choose_model(cfg, torch.device(device))
    model.load_state_dict(torch.load(p, map_location=device, weights_only=True))
    model.hurdle_nb_theta = cfg.get("hurdle_nb_theta")
    model.hurdle_lognormal_sigma = cfg.get("hurdle_lognormal_sigma")
    model.to(device).eval()      # eval(): no dropout, and BN buffers are NOT touched
    # DECIDED, not inherited (raised by the views-models session 2026-10-12). views-models#549's
    # collection is degraded-not-fatal when `.pt.sha256` is missing: it keeps the weights and
    # PROVENANCE then reads `sha256: not recorded`. A blanket digest comparison would turn that
    # deliberate degradation into a VOID here. It is ACCEPTED WITH A WARNING instead, because:
    #   * the digest is computed from the file above either way, so the C-317 divergence record
    #     against the delivered d92dcff4… is preserved regardless of the sidecar;
    #   * the only thing actually lost is transit-corruption detection, and a corrupted state dict
    #     would fail `load_state_dict` on keys or shapes before any measurement ran;
    #   * refusing a trained 300-lesson model over a missing metadata file is disproportionate —
    #     the same reasoning that justifies the upstream degraded path.
    # A sidecar that EXISTS and DISAGREES is a different thing and stays fatal (checked above).
    prov_sha = prov.get("sha256", "").strip()
    if prov_sha in ("", "not recorded"):
        print(
            "⚠️  PROVENANCE carries no sha256 (`not recorded`) — accepted, not VOID. Integrity "
            "against the pod's own record cannot be checked; the digest printed below is computed "
            "here. See vehicle.py for why this is accepted rather than refused."
        )
    elif prov_sha != sha:
        raise SystemExit(
            f"VOID (F6): PROVENANCE records sha256 {prov_sha[:16]}… but the file's own digest is "
            f"{sha[:16]}…. The artifact does not match its own provenance record."
        )
    print(f"vehicle: {p.name}  sha256 {sha[:16]}…  (NOT the delivered weights — C-317)")
    print(f"  lessons {lessons} · run_type {prov.get('run_type', '?')} · "
          f"git {prov.get('git', '?')} · trained {prov.get('trained', '?')}")
    print(
        "⚠️  REGION IS RECORDED AS ENFORCED, NOT MEASURED. PROVENANCE says "
        f"{prov.get('region', '?')!r} — which records that pod_run_model.sh:279-282 refuses any "
        "other value, NOT a read-back of the grid the weights saw. Nothing in a .pt or its "
        "sidecar records the region, and the net is fully convolutional, so this remains weaker "
        "than a measurement. The four known africa artifacts are additionally denied by hash."
    )
    return model, cfg, sha


def pull(months: tuple[int, int], cache: str | Path):
    """Fresh datafactory pull of all three targets at REGION='land', cached to `cache`."""
    p = Path(cache)
    if p.exists():
        return pd.read_parquet(p).reset_index()
    from datafactory_query import load_dataset
    from datafactory_query.defaults import DEFAULT_REMOTE

    df = load_dataset(region="land", start=months[0], end=months[1], features=TARGETS,
                      output_format="dataframe", data_dir=DEFAULT_REMOTE.zarr_url)
    df.to_parquet(p)
    return df.reset_index()


def build_volume(
    df: pd.DataFrame, months: list[int]
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """-> (volume [1, T, 3, 360, 720] in log1p space, land_mask [360,720] bool, pg_grid [360,720]).

    log1p because the pipeline trains in log1p space and the family's `nll` recovers raw counts
    itself via `to_raw_counts` (count_target_bridge). Non-land cells are left at 0, which is what
    the production raster does — `config_queryset.py:27`, "rows/cols with no land are zero".
    """
    cells = np.sort(df.priogrid_id.unique())
    r, c = grid_rowcol(cells)
    land = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    land[r - 1, c - 1] = True
    pg_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int64)
    pg_grid[r - 1, c - 1] = cells

    vol = np.zeros((1, len(months), len(TARGETS), GRID_ROWS, GRID_COLS), dtype=np.float32)
    bym = {m: g for m, g in df.groupby("month_id")}
    for ti, m in enumerate(months):
        g = bym[m].set_index("priogrid_id").reindex(cells)
        for ci, t in enumerate(TARGETS):
            vol[0, ti, ci, r - 1, c - 1] = np.log1p(g[t].to_numpy(np.float64))
    return torch.from_numpy(vol), land, pg_grid


def sb_params(model_out_reg: torch.Tensor, n_params: int = 2) -> torch.Tensor:
    """Slice sb's family params from `out.reg` [B, n_reg*n_params, H, W] -> [B, H, W, n_params].

    Mirrors training_engine.py:686 exactly (ADR-067 / C-207).
    """
    return model_out_reg[:, SB * n_params:(SB + 1) * n_params, :, :].permute(0, 2, 3, 1)


def active_cells_top100(df: pd.DataFrame, lo: int, hi: int, n: int = 100) -> np.ndarray:
    """The prereg's cell set (ii): top-n by OBSERVED positive rate in [lo, hi], tiebreak total."""
    w = df[(df.month_id >= lo) & (df.month_id <= hi)]
    piv = w.pivot(index="priogrid_id", columns="month_id", values="ged_sb_best")
    k = (piv.to_numpy() > 0).sum(1) / piv.shape[1]
    tot = piv.to_numpy().sum(1)
    return piv.index.to_numpy()[np.lexsort((-tot, -k))[:n]]
