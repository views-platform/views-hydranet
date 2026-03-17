"""
Standalone DataFetcher component for the HydraNet pipeline.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.files.utils import read_dataframe

from views_hydranet.utils.utils_logging import log_data_load_report

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Component responsible for retrieving raw VIEWS DataFrames from disk.

    This class decouples the data ingestion process from model-specific
    transformations, allowing for independent inspection and validation
    of the raw data state.
    """

    def __init__(self, path_raw: str | Path, config: Dict[str, Any]) -> None:
        """
        Initializes with the physical data path and the active configuration.
        """
        self.path_raw = path_raw
        self.config = config

    def fetch_df(self) -> pd.DataFrame:
        """
        Loads the DataFrame for the current run_type defined in the config.

        Returns:
            pd.DataFrame: The raw data as fetched from the pipeline output.
        """
        partition = self.config["run_type"]
        df_ext = PipelineConfig.dataframe_format
        path_raw_file = os.path.join(str(self.path_raw), f"{partition}_viewser_df{df_ext}")

        print("")
        logger.info(f"DataFetcher: Loading {partition} from {path_raw_file}")
        print("")

        df = read_dataframe(path_raw_file)

        log_data_load_report(partition, path_raw_file, df)

        return df

    @staticmethod
    def standardize_raw_df(df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """
        Standardizes a raw VIEWS DataFrame by enforcing strict index structure from config.

        Requirements:
        - Must be a pandas MultiIndex.
        - Level names must match config["index_names"].

        Raises:
            ValueError: If the index structure does not match exactly.
        """

        # 1. Enforce MultiIndex
        if not isinstance(df.index, pd.MultiIndex):
            err_msg = f"[CRITICAL DATA ERROR] Expected MultiIndex, got {type(df.index)}"

            logger.error(err_msg)

            raise ValueError(err_msg)

        # 2. Enforce Level Names and Order from Config (ADR 017 Section 1.2)
        try:
            expected_names = config["index_names"]
        except KeyError:
            err_msg = (
                "DataFetcher Contract Violation: 'index_names' missing from config.\n"
                "To comply with ADR 017, you must explicitly define the MultiIndex levels."
            )

            logger.error(err_msg)

            raise KeyError(err_msg)

        actual_names = list(df.index.names)

        if actual_names[: len(expected_names)] != expected_names:
            err_msg = (
                f"[CRITICAL DATA ERROR] Index Contract Violation!\n"
                f"Expected levels: {expected_names}\n"
                f"Actual levels:   {actual_names}"
            )

            logger.error(err_msg)

            raise ValueError(err_msg)

        # 3. Structural Normalization
        df_flat = df.reset_index()

        # 4. Sort-Order Assertion (Hypothesis 5 Probe)
        # We must ensure time is monotonic for the VolumeHandler to work correctly.
        time_col = config.get("time_col", "month_id")
        id_col = config.get("id_col", "priogrid_gid")

        if time_col in df_flat.columns:
            if not df_flat[time_col].is_monotonic_increasing:
                logger.warning(
                    f"⚠️ DataFetcher: DataFrame is NOT sorted by {time_col}. "
                    "This may cause 'Random Number' scrambling in VolumeHandler."
                )
                # We enforce sort to be safe
                df_flat = df_flat.sort_values(by=[time_col, id_col])
                logger.info(f"✅ DataFetcher: Enforced sort by [{time_col}, {id_col}].")

        # 5. Geo-Sanitization (Ocean Removal)
        # We only keep cells that have a valid ID.
        if id_col not in df_flat.columns:
            err_msg = f"DataFetcher Sanitization Failed: '{id_col}' not found after reset_index."
            logger.error(err_msg)
            raise KeyError(err_msg)

        # Drop rows where ID is 0 or NaN (Ocean)
        df_out = df_flat[df_flat[id_col] > 0].copy()

        # 6. Ontological Standardization (ADR 046)
        # Ensure the DataFrame matches the instructional blueprint
        return DataFetcher.apply_blueprint(df_out, config)

    # Cross-ref: VolumeHandler._execute_derivations() (volume_handler.py)
    # This path SKIPS if source missing; volume path RAISES.
    # See tests/test_derivation_parity.py for parity guard.
    @staticmethod
    def apply_blueprint(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Executes the instructional derivations (ADR 046) on a DataFrame.
        This ensures that DataFrames used for training and evaluation contain
        the derived signals (e.g., binary targets) defined in the blueprint.
        """
        df_out = df.copy()
        derivations = config.get("derivations", {})

        for op, instructions in derivations.items():
            for instr in instructions:
                src_name = instr.get("from")
                dst_name = instr.get("to")

                if not all([src_name, dst_name]):
                    err_msg = (
                        f"DataFetcher Blueprint Error: Missing mandatory keys (from, to) "
                        f"in {op} instr: {instr}"
                    )
                    logger.error(err_msg)
                    raise KeyError(err_msg)

                if src_name not in df_out.columns:
                    # If the source is missing, we can't derive.
                    logger.debug(
                        f"DataFetcher Blueprint: Source '{src_name}' missing. "
                        "Skipping derivation."
                    )
                    continue

                if op == "binary":
                    if "threshold" not in instr:
                        err_msg = (
                            f"DataFetcher Blueprint Error: 'binary' op requires "
                            f"explicit 'threshold' key in {instr}"
                        )
                        logger.error(err_msg)
                        raise KeyError(err_msg)

                    threshold = instr["threshold"]
                    df_out[dst_name] = (df_out[src_name] > threshold).astype(float)
                else:
                    err_msg = (
                        f"DataFetcher Blueprint Error: Operation '{op}' not implemented."
                    )
                    logger.error(err_msg)
                    raise NotImplementedError(err_msg)

        return df_out
