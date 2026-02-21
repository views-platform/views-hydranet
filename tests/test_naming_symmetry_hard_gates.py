import numpy as np
import pandas as pd
import pytest

from views_hydranet.utils.volume_handler import VolumeHandler

# NAMING CONFIG
CFG = {
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "identity_cols": ["month_id", "priogrid_gid"],
    "features": ["lr_sb", "lr_ns"],
    "row_offset": 0,
    "col_offset": 0,
    "height": 4,
    "width": 4,
    # ADR 046 Compliance
    "transformations": {"identity": ["lr_sb", "lr_ns"]},
    "derivations": {
        "binary": [
            {"from": "lr_sb", "to": "by_sb", "threshold": 0},
            {"from": "lr_ns", "to": "by_ns", "threshold": 0},
        ]
    },
}


class TestNamingEngineFalsification:
    """Popperian battery to falsify the Naming Engine across all dimensions."""

    @pytest.fixture
    def history_scaffold(self):
        df = pd.DataFrame(
            {
                "month_id": [100],
                "priogrid_gid": [1],
                "row": [0],
                "col": [0],
                "lr_sb": [10.0],
                "lr_ns": [5.0],
            }
        )
        return VolumeHandler.from_df(df, CFG)

    def test_gate_1_to_4_naming_symmetry(self, history_scaffold):
        """Gates 1-4: Verify naming and value types for Point vs Stochastic."""
        np.random.seed(
            42
        )  # Fix seed: naming test cares about types, not values, but reproducibility matters

        # --- SCENARIO A: POINT (4D) ---
        # [T=1, H=4, W=4, C=4] (sb_reg, ns_reg, sb_prob, ns_prob)
        posterior_4d = np.random.rand(1, 4, 4, 4)
        pred_vh_4d = history_scaffold.wrap_predictions(
            posterior_4d, target_names=["lr_sb", "lr_ns", "by_sb", "by_ns"]
        )
        df_4d = pred_vh_4d.to_evaluation_df(history=history_scaffold, start_idx=0)

        # Gate 1: Point Regression
        assert "pred_lr_sb" in df_4d.columns
        assert isinstance(df_4d["pred_lr_sb"].iloc[0], (float, np.float32, np.float64))

        # Gate 2: Point Classification
        assert "pred_by_sb" in df_4d.columns
        assert isinstance(df_4d["pred_by_sb"].iloc[0], (float, np.float32, np.float64))

        # --- SCENARIO B: STOCHASTIC (5D) ---
        # [T=1, H=4, W=4, C=4, S=5]
        posterior_5d = np.random.rand(1, 4, 4, 4, 5)
        pred_vh_5d = history_scaffold.wrap_predictions(
            posterior_5d, target_names=["lr_sb", "lr_ns", "by_sb", "by_ns"]
        )
        df_5d = pred_vh_5d.to_evaluation_df(history=history_scaffold, start_idx=0)

        # Gate 3: Stochastic Regression
        assert "pred_lr_sb" in df_5d.columns
        assert isinstance(df_5d["pred_lr_sb"].iloc[0], list)
        assert len(df_5d["pred_lr_sb"].iloc[0]) == 5

        # Gate 4: Stochastic Classification
        assert "pred_by_sb" in df_5d.columns
        assert isinstance(df_5d["pred_by_sb"].iloc[0], list)
        assert len(df_5d["pred_by_sb"].iloc[0]) == 5

    def test_gate_5_to_8_integrity_and_collision(self, history_scaffold):
        """Gates 5-8: Verify actuals protection, cleanup, and multi-target symmetry."""

        posterior = np.zeros((1, 4, 4, 4))
        pred_vh = history_scaffold.wrap_predictions(
            posterior, target_names=["lr_sb", "lr_ns", "by_sb", "by_ns"]
        )
        df = pred_vh.to_evaluation_df(history=history_scaffold, start_idx=0)

        cols = df.columns.tolist()

        # Gate 5: Actual Preservation
        # The ground truth  'lr_sb' must exist alongside the prediction
        assert "lr_sb" in cols
        assert "lr_ns" in cols
        # Value must match input (10.0)
        assert df["lr_sb"].iloc[0] == 10.0

        # Gate 6: Internal Cleanup
        # No internal tokens should ever reach the researcher
        assert not any("_INTERNAL_" in c for c in cols)
        assert not any("ACTUAL_INTERNAL_" in c for c in cols)

        # Gate 7: Prefix Constancy
        # Every prediction MUST start with pred_
        preds = [c for c in cols if c.startswith("pred_")]
        assert len(preds) == 4  # (lr_sb, by_sb, lr_ns, by_ns)
        assert all(c.startswith("pred_") for c in preds)

        # Gate 8: Multi-Target Symmetry
        # Both sb and ns must have their triplet (Actual, Prediction, Binary)
        for target in ["sb", "ns"]:
            assert f"lr_{target}" in cols  # Linear Actual
            assert f"by_{target}" in cols  # Binary Actual
            assert f"pred_lr_{target}" in cols  # Linear Prediction
            assert f"pred_by_{target}" in cols  # Binary Prediction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
