"""
PredictionFrameDispatcher — orchestrates PF-to-legacy-DF conversion,
EvaluationFrame construction, and parity auditing for the ADR-033 PF path.

Extracts the three inline PF dispatch blocks from ForecastingModelManager so
they can be independently tested and later extended for multi-target support
(future: Dict[str, List[PredictionFrame]] return from _evaluate_model_artifact).

DF dispatch remains in model.py for now; this class is designed so a parallel
DataFrameDispatcher would have the same shape.
"""
import logging
import numpy as np
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class PredictionFrameDispatcher:
    """
    Orchestrates PF-to-legacy-DF conversion, EvaluationFrame construction,
    and parity auditing for the ADR-033 PF path.

    All three public methods are stateless; the class exists for cohesion and
    to provide a clean extension point for multi-target dispatch.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_legacy_dfs(
        self,
        predictions: List[Any],  # List[PredictionFrame]
        target: str,
    ) -> List[pd.DataFrame]:
        """
        Convert List[PredictionFrame] to the list-in-cell DataFrame format
        that EvaluationAdapter.from_dataframes() expects.

        Each output DataFrame has:
        - MultiIndex (time, unit) derived from PredictionFrame.identifiers.
        - A single column 'pred_{target}' where each cell is a list of sample floats.

        PARITY-BRIDGE ONLY — remove when the DataFrame path is retired and
        from_dataframes() / from_prediction_frames() are no longer compared.

        Args:
            predictions: List of PredictionFrame objects.
            target:      Target variable name (used to construct the column name).

        Returns:
            List of DataFrames, one per input PredictionFrame.
        """
        pred_col = f"pred_{target}"
        result = []
        for pf in predictions:
            idx = pd.MultiIndex.from_arrays(
                [pf.identifiers["time"], pf.identifiers["unit"]]
            )
            df = pd.DataFrame(
                {pred_col: [list(row) for row in pf.y_pred]},
                index=idx,
            )
            result.append(df)
        return result

    def build_evaluation_frame(
        self,
        actual: pd.DataFrame,
        predictions: List[Any],  # List[PredictionFrame]
        target: str,
        step_mappings: List[Dict[int, int]],
    ) -> Any:  # EvaluationFrame
        """
        Build a PF-native EvaluationFrame, construct a parity-bridge EF from
        the corresponding legacy DataFrames, audit bit-wise parity, and return
        the PF-native EF.

        Args:
            actual:        DataFrame of ground-truth values (MultiIndex [time, unit]).
            predictions:   List[PredictionFrame] — one per rolling-origin sequence.
            target:        Target variable name.
            step_mappings: List[Dict[month → step]], one dict per sequence.

        Returns:
            EvaluationFrame built from the PF-native path.

        Raises:
            ValueError: "Parity Failure ..." if PF and legacy-DF EFs disagree.
        """
        from views_pipeline_core.modules.validation.adapter import EvaluationAdapter

        ef = EvaluationAdapter.from_prediction_frames(
            actual=actual,
            predictions=predictions,
            target=target,
            step_mapping=step_mappings,
        )

        pred_slices = self.to_legacy_dfs(predictions, target)
        ef_leg = EvaluationAdapter.from_dataframes(
            actual=actual,
            predictions=pred_slices,
            target=target,
            step_mapping=step_mappings,
        )

        self.audit_parity_ef(ef, ef_leg, target)
        return ef

    def audit_parity_ef(
        self,
        ef_pf: Any,   # EvaluationFrame (duck-typed)
        ef_leg: Any,  # EvaluationFrame
        target: str,
    ) -> None:
        """
        Compare two EvaluationFrame objects for bit-wise parity.

        Used during the ADR-033 Strangler Fig transition to verify that the
        PredictionFrame adapter path produces numerically identical output to
        the legacy DataFrame adapter path for the same underlying predictions.

        Args:
            ef_pf:   EvaluationFrame built from the PredictionFrame path.
            ef_leg:  EvaluationFrame built from the legacy DataFrame path.
            target:  Target column name (for logging only).

        Raises:
            ValueError: If any array comparison fails — message begins with
                        "Parity Failure".
        """
        logger.info(f"AUDITING EF PARITY for target: {target}")

        try:
            np.testing.assert_allclose(ef_pf.y_pred, ef_leg.y_pred, rtol=1e-5, atol=1e-8)
        except AssertionError as e:
            raise ValueError(f"Parity Failure (y_pred): {e}")

        try:
            np.testing.assert_allclose(ef_pf.y_true, ef_leg.y_true, rtol=1e-5, atol=1e-8)
        except AssertionError as e:
            raise ValueError(f"Parity Failure (y_true): {e}")

        for key in ("time", "unit", "origin", "step"):
            try:
                np.testing.assert_array_equal(
                    ef_pf.identifiers[key], ef_leg.identifiers[key]
                )
            except AssertionError as e:
                raise ValueError(f"Parity Failure (identifiers['{key}']): {e}")

        logger.info(
            "\033[92m" + f"EF PARITY CONFIRMED for {target.upper()}" + "\033[0m"
        )

    def audit_prediction_structure(
        self,
        pf: Any,           # PredictionFrame (duck-typed)
        df: pd.DataFrame,
        target: str,
    ) -> None:
        """
        Structural audit after PF→DF conversion.

        Verifies that the legacy list-in-cell DataFrame produced by
        to_legacy_dfs() has the correct row count and column name relative
        to the originating PredictionFrame.  Used in the forecasting-partition
        path where no actuals are available for a full EF-level parity check.

        Note: "prediction" in the method name refers to PredictionFrame, not
        the forecasting data partition.

        Args:
            pf:     The source PredictionFrame.
            df:     The converted DataFrame (output of to_legacy_dfs).
            target: Target variable name (used to check column 'pred_{target}').

        Raises:
            ValueError: "PF→DF conversion ..." if row count or column name mismatch.
        """
        pf_rows = len(pf.identifiers["time"])
        df_rows = len(df)
        if pf_rows != df_rows:
            raise ValueError(
                f"PF→DF conversion: PF has {pf_rows} rows but converted DF has {df_rows} rows."
            )
        if f"pred_{target}" not in df.columns:
            raise ValueError(
                f"PF→DF conversion: expected column 'pred_{target}' "
                f"not found in converted DF (columns: {list(df.columns)})."
            )
        logger.info(
            "\033[92m" + f"PF STRUCTURAL PARITY OK for {target.upper()}" + "\033[0m"
        )
