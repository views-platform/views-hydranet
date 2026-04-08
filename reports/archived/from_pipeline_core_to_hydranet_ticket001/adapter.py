import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from views_evaluation.evaluation.evaluation_frame import EvaluationFrame

logger = logging.getLogger(__name__)

class EvaluationAdapter:
    """
    Adapter to convert Pandas DataFrames into the native EvaluationFrame.
    
    This class 'knows' about Pandas, allowing the rest of the core
    to remain pure.
    """
    
    @staticmethod
    def from_dataframes(
        actual: pd.DataFrame,
        predictions: List[pd.DataFrame],
        target: str,
        step_mapping: Optional[Union[Dict[int, int], List[Dict[int, int]]]] = None,
    ) -> EvaluationFrame:
        """
        Convert the current List[DataFrame] structure into a single EvaluationFrame.

        Args:
            actual: DataFrame with MultiIndex [time, unit]
            predictions: List of DataFrames with MultiIndex [time, unit]
            target: The name of the target column
            step_mapping: Optional step assignment authority.
                - Dict[int, int]: a single mapping applied to all sequences (single-origin).
                - List[Dict[int, int]]: one mapping per sequence for rolling-origin evaluation
                  where each sequence is anchored at a different origin month.
                - None: steps are inferred positionally (Legacy fallback).
        """
        
        all_y_true = []
        all_y_pred = []
        all_times = []
        all_units = []
        all_origins = []
        all_steps = []
        
        pred_col = f"pred_{target}"
        
        if target not in actual.columns:
            raise KeyError(f"Target column '{target}' not found in actuals.")

        if not predictions:
            # Align with legacy expected error message
            raise ValueError("No objects to concatenate")

        # GUARD: mapping-count — step_mapping list must have one entry per prediction sequence.
        # An IndexError on step_mapping[i] would be cryptic; make it explicit here.
        if isinstance(step_mapping, list) and len(step_mapping) != len(predictions):
            raise ValueError(
                f"step_mapping list length ({len(step_mapping)}) must equal the number "
                f"of prediction sequences ({len(predictions)}). Each sequence requires "
                f"its own explicit origin-anchored mapping."
            )

        for i, df in enumerate(predictions):
            # Resolve the mapping for THIS sequence (ADR-031 compliance).
            # Done before intersection so the window integrity check has access to it.
            if isinstance(step_mapping, list):
                seq_mapping = step_mapping[i]   # rolling-origin: one dict per sequence
            else:
                seq_mapping = step_mapping      # single dict or None: backward-compatible

            # GUARD: window-integrity — every prediction month must fall within the declared step_mapping.
            # Prove that EVERY month the model produced is inside the declared step window,
            # regardless of whether that month appears in the actuals.
            # This closes the pre-intersection blindspot: months dropped by the actuals
            # intersection are still checked here, before the intersection occurs.
            #
            # Formal guarantee (see proof in docstring):
            #   ∀ m ∈ D_i: m ∈ keys(seq_mapping)  →  seq_mapping[m] = m - (base_origin + i)
            if seq_mapping is not None:
                pred_months = set(df.index.get_level_values(0).unique())
                rogue_months = pred_months - set(seq_mapping.keys())
                if rogue_months:
                    raise ValueError(
                        f"Sequence {i}: prediction contains month(s) {sorted(rogue_months)} "
                        f"that are not in the declared step_mapping window "
                        f"(expected months: {sorted(seq_mapping.keys())[:5]}{'...' if len(seq_mapping) > 5 else ''}). "
                        f"This indicates that the declared base_origin does not match the "
                        f"model's actual forecast origin for this sequence."
                    )

            # 1. Align/Match Actuals (duplicated logic from EvaluationManager)
            common_idx = actual.index.intersection(df.index)
            if common_idx.empty:
                # Warn rather than silently drop: a sequence with zero overlap is unexpected
                # during rolling-origin evaluation but is not an error in itself (e.g. the
                # forecast window is entirely beyond the actuals horizon).
                logger.warning(
                    f"Sequence {i}: no overlap between prediction index and actuals index. "
                    f"This sequence contributes zero rows to the EvaluationFrame."
                )
                continue

            matched_pred = df.loc[common_idx]
            matched_actual = actual.loc[common_idx, target]
            
            # 2. Extract Data
            # Note: We assume all cells have the same number of samples
            # This is where we explode the 'list-in-cell'
            sample_lists = matched_pred[pred_col].tolist()
            
            # ADR-012: Validate rectangular samples
            lengths = [len(x) if isinstance(x, (list, np.ndarray)) else 1 for x in sample_lists]
            if len(set(lengths)) > 1:
                # Align with legacy expected error message
                raise ValueError(
                    f"Inconsistent list lengths in sample evaluation. "
                    f"Found lengths {set(lengths)}"
                )

            samples = np.array(sample_lists)
            if samples.ndim == 1: # Point forecasts
                samples = samples.reshape(-1, 1)
            
            n_rows = len(matched_actual)
            
            # Legacy Actuals might be list-like (e.g. [0.1])
            actual_vals = matched_actual.values
            if actual_vals.dtype == object:
                # Coerce to scalars
                actual_vals = np.array([
                    x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x 
                    for x in actual_vals
                ])

            all_y_true.append(actual_vals)
            all_y_pred.append(samples)
            
            # 3. Extract Identifiers
            times = matched_pred.index.get_level_values(0).values
            units = matched_pred.index.get_level_values(1).values
            
            # ADR-012: No NaNs in identifiers
            if np.any(pd.isna(times)):
                raise ValueError(f"NaN detected in 'time' index level of sequence {i}.")
            if np.any(pd.isna(units)):
                raise ValueError(f"NaN detected in 'unit' index level of sequence {i}.")

            all_times.append(times)
            all_units.append(units)
            
            # 4. Synthesize Origin and Step
            # Origin is the list index
            all_origins.append(np.full(n_rows, i))

            # seq_mapping was resolved at the top of the loop (before window integrity check)

            # Step assignment
            if seq_mapping is not None:
                steps = []
                for t in times:
                    if t not in seq_mapping:
                        raise ValueError(
                            f"Month ID {t} not found in step_mapping for sequence {i}."
                        )
                    steps.append(seq_mapping[t])
                steps = np.array(steps)
            else:
                # Step is positional lead-time per unique month in the sequence (Legacy)
                unique_times = matched_pred.index.get_level_values(0).unique()
                time_to_step = {t: step_idx + 1 for step_idx, t in enumerate(unique_times)}
                steps = np.array([time_to_step[t] for t in times])
            
            all_steps.append(steps)
            
        if not all_y_true:
            # ADR-013: Fail-Loud on zero overlap
            raise ValueError("need at least one array to concatenate")

        # ADR-012: Ensure all sequences have consistent sample counts
        sample_counts = [y.shape[1] for y in all_y_pred]
        if len(set(sample_counts)) > 1:
            raise ValueError(
                "Mix of evaluation types detected: some sequences contain point forecasts, others contain samples. "
                "Please ensure all sequences are consistent in their evaluation type."
            )

        return EvaluationFrame(
            y_true=np.concatenate(all_y_true),
            y_pred=np.concatenate(all_y_pred),
            identifiers={
                'time': np.concatenate(all_times),
                'unit': np.concatenate(all_units),
                'origin': np.concatenate(all_origins),
                'step': np.concatenate(all_steps),
            },
            metadata={'target': target}
        )

    @staticmethod
    def from_prediction_frame(
        actual: pd.DataFrame,
        prediction_frame: Any,  # Avoid circular import, type is PredictionFrame
        target: str,
        step_mapping: Dict[int, int] = None,
    ) -> EvaluationFrame:
        """
        Convert a PredictionFrame into an EvaluationFrame by aligning with actuals.
        
        Args:
            actual: DataFrame with MultiIndex [time, unit]
            prediction_frame: PredictionFrame containing arrays
            target: The name of the target column in actuals
            step_mapping: Optional explicit lead-time mapping
        """
        if target not in actual.columns:
            raise KeyError(f"Target column '{target}' not found in actuals.")

        # GUARD: window-integrity — every prediction month must fall within the declared step_mapping.
        if step_mapping is not None:
            pred_months = set(prediction_frame.identifiers['time'].tolist())
            rogue_months = pred_months - set(step_mapping.keys())
            if rogue_months:
                raise ValueError(
                    f"Prediction contains month(s) {sorted(rogue_months)} that are not "
                    f"in the declared step_mapping window "
                    f"(expected months: {sorted(step_mapping.keys())[:5]}"
                    f"{'...' if len(step_mapping) > 5 else ''}). "
                    f"This indicates that the declared base_origin does not match the "
                    f"model's actual forecast origin."
                )

        # 1. Alignment (Intersection)
        # We must align the prediction_frame arrays with the actuals index.
        # Since PredictionFrame has flat arrays, we'll use pandas to perform 
        # the intersection efficiently.
        
        # Create a temporary index from PF identifiers to perform intersection
        pf_index = pd.MultiIndex.from_arrays(
            [prediction_frame.identifiers['time'], prediction_frame.identifiers['unit']],
            names=actual.index.names
        )
        
        common_idx = actual.index.intersection(pf_index)
        if common_idx.empty:
            raise ValueError("need at least one array to concatenate")

        # 2. Extract matched data
        # We need to find the integer locations in pf_index that match common_idx
        # This preserves the "Join" semantics exactly.
        pf_locs = pf_index.get_indexer(common_idx)
        
        y_pred = prediction_frame.y_pred[pf_locs]
        y_true = actual.loc[common_idx, target].values
        
        # Coerce legacy actuals if needed
        if y_true.dtype == object:
            y_true = np.array([
                x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x 
                for x in y_true
            ])

        # 3. Identifiers
        times = prediction_frame.identifiers['time'][pf_locs]
        units = prediction_frame.identifiers['unit'][pf_locs]
        
        # 4. Synthesize Origin and Step
        origin = np.zeros(len(y_true), dtype=int) # Single sequence for PredictionFrame
        
        if step_mapping is not None:
            steps = np.array([step_mapping[t] for t in times])
        else:
            # Positional inference (consistent with legacy from_dataframes)
            unique_times = pd.Series(times).unique()
            time_to_step = {t: i + 1 for i, t in enumerate(unique_times)}
            steps = np.array([time_to_step[t] for t in times])

        return EvaluationFrame(
            y_true=y_true,
            y_pred=y_pred,
            identifiers={
                'time': times,
                'unit': units,
                'origin': origin,
                'step': steps,
            },
            metadata={'target': target}
        )

    @staticmethod
    def from_prediction_frames(
        actual: pd.DataFrame,
        predictions: List[Any],  # List[PredictionFrame]
        target: str,
        step_mapping: Optional[List[Dict[int, int]]] = None,
    ) -> EvaluationFrame:
        """
        Convert a List[PredictionFrame] (one per evaluation sequence) into a single
        EvaluationFrame. Mirrors from_dataframes() exactly but uses the dense
        PredictionFrame arrays directly, bypassing list-in-cell explosion.

        Args:
            actual: DataFrame with MultiIndex [time, unit]
            predictions: List of PredictionFrames, one per rolling-origin sequence.
                Each PredictionFrame must have identifiers["time"] and identifiers["unit"].
            target: The name of the target column in actuals.
            step_mapping: List of dicts, one per sequence (len must match predictions).
                Each dict maps month_id → step. Required for rolling-origin evaluation;
                None falls back to positional inference (legacy).
        """
        if target not in actual.columns:
            raise KeyError(f"Target column '{target}' not found in actuals.")

        if not predictions:
            raise ValueError("No objects to concatenate")

        # GUARD: mapping-count — step_mapping list must have one entry per prediction sequence.
        if isinstance(step_mapping, list) and len(step_mapping) != len(predictions):
            raise ValueError(
                f"step_mapping list length ({len(step_mapping)}) must equal the number "
                f"of prediction sequences ({len(predictions)}). Each sequence requires "
                f"its own explicit origin-anchored mapping."
            )

        all_y_true   = []
        all_y_pred   = []
        all_times    = []
        all_units    = []
        all_origins  = []
        all_steps    = []

        for i, pf in enumerate(predictions):
            seq_mapping = step_mapping[i] if isinstance(step_mapping, list) else step_mapping

            # GUARD: window-integrity — every prediction month must fall within the declared step_mapping.
            if seq_mapping is not None:
                pred_months = set(pf.identifiers['time'].tolist())
                rogue_months = pred_months - set(seq_mapping.keys())
                if rogue_months:
                    raise ValueError(
                        f"Sequence {i}: prediction contains month(s) {sorted(rogue_months)} "
                        f"that are not in the declared step_mapping window "
                        f"(expected months: {sorted(seq_mapping.keys())[:5]}"
                        f"{'...' if len(seq_mapping) > 5 else ''}). "
                        f"This indicates that the declared base_origin does not match the "
                        f"model's actual forecast origin for this sequence."
                    )

            # Build a temporary MultiIndex from PF identifiers for alignment.
            pf_index = pd.MultiIndex.from_arrays(
                [pf.identifiers['time'], pf.identifiers['unit']],
                names=actual.index.names,
            )

            common_idx = actual.index.intersection(pf_index)
            if common_idx.empty:
                logger.warning(
                    f"Sequence {i}: no overlap between prediction index and actuals index. "
                    f"This sequence contributes zero rows to the EvaluationFrame."
                )
                continue

            pf_locs = pf_index.get_indexer(common_idx)

            y_pred_seq = pf.y_pred[pf_locs]
            y_true_seq = actual.loc[common_idx, target].values

            if y_true_seq.dtype == object:
                y_true_seq = np.array([
                    x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x
                    for x in y_true_seq
                ])

            times = pf.identifiers['time'][pf_locs]
            units = pf.identifiers['unit'][pf_locs]

            if np.any(pd.isna(times)):
                raise ValueError(f"NaN detected in 'time' identifiers of sequence {i}.")
            if np.any(pd.isna(units)):
                raise ValueError(f"NaN detected in 'unit' identifiers of sequence {i}.")

            all_y_true.append(y_true_seq)
            all_y_pred.append(y_pred_seq)
            all_times.append(times)
            all_units.append(units)
            all_origins.append(np.full(len(y_true_seq), i))

            if seq_mapping is not None:
                steps = np.array([seq_mapping[t] for t in times])
            else:
                unique_times = pd.Series(times).unique()
                time_to_step = {t: idx + 1 for idx, t in enumerate(unique_times)}
                steps = np.array([time_to_step[t] for t in times])
            all_steps.append(steps)

        if not all_y_true:
            raise ValueError("need at least one array to concatenate")

        sample_counts = [y.shape[1] for y in all_y_pred]
        if len(set(sample_counts)) > 1:
            raise ValueError(
                "Inconsistent sample counts across PredictionFrame sequences: "
                f"{set(sample_counts)}."
            )

        return EvaluationFrame(
            y_true=np.concatenate(all_y_true),
            y_pred=np.concatenate(all_y_pred),
            identifiers={
                'time':   np.concatenate(all_times),
                'unit':   np.concatenate(all_units),
                'origin': np.concatenate(all_origins),
                'step':   np.concatenate(all_steps),
            },
            metadata={'target': target}
        )

