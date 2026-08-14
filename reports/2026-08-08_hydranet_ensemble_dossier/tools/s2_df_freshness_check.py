"""S2 Tier-A (transitivity path): a FRESH datafactory pull vs the frozen v2 truth.

The 3 viewser models (pink_pirate, blue_stranger, purple_alien) use a queryset BYTE-IDENTICAL to
violet_visitor's pre-migration queryset (same ged_*_best_sum_nokgi, same africa_me_legacy) — so they
pull the same dataset violet already Tier-A-PASSed vs viewser (2026-07-28, fresh). The only genuinely
NEW check is whether the datafactory source is still live + stable. This pulls FRESH and compares to
the frozen v2 truth (itself a 2026-07-28 datafactory pull) using the Tier-A scorecard: cell-set,
coverage, maxima, drift within band ⇒ the source is stable ⇒ transitivity holds ⇒ safe to migrate.
"""
import sys

import pandas as pd

MIG = "/home/simon/Documents/scripts/views_platform/views-hydranet/reports/2026-07-28_datafactory_migration_dossier/tools"
sys.path.insert(0, MIG)
import tier_a_parity as T  # noqa: E402

V2_TRUTH = f"{MIG}/v2_truth/calibration_datafactory_df.parquet"

frozen = pd.read_parquet(V2_TRUTH)  # the reference (already lr_*_best renamed)
print(f"frozen v2 truth: {frozen.shape} cols={list(frozen.columns)}")
print("FRESH datafactory pull (region=africa_me_legacy, 121-504, NOT cached)...")
fresh = T.fresh_pull(region="africa_me_legacy", start=121, end=504)
print(f"fresh pull: {fresh.shape} | remote last_valid_month_id={fresh.attrs.get('last_valid_month_id')}")

# frozen truth is the reference (lr_* names); fresh is the datafactory (ged_* -> lr_* via feature_map)
sc = T.parity_scorecard(frozen, fresh, T.DEFAULT_FEATURE_MAP)
verdict = T.evaluate_falsifiers(sc)
T._print_report(sc, verdict)
print(f"\nS2 datafactory-freshness verdict: {'STABLE (PASS)' if verdict['passed'] else 'DRIFTED (FAIL)'}")
sys.exit(0 if verdict["passed"] else 2)
