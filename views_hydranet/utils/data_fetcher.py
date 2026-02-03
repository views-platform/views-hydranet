"""
Standalone DataFetcher component for the HydraNet pipeline.
"""

import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.files.utils import read_dataframe


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
        df_ext = PipelineConfig().dataframe_format
        path_raw_file = os.path.join(
            str(self.path_raw), f"{partition}_viewser_df{df_ext}"
        )

        print(f"!!! DataFetcher: Loading {partition} from {path_raw_file}")
        df = read_dataframe(path_raw_file)

        # Explicit Debugging Block
        print("\n" + "=" * 60)
        print("!!! DEBUG: DataFetcher - Ingestion Complete")
        print(f"!!! Partition: {partition}")
        print(f"!!! Columns:   {df.columns.tolist()}")
        print("=" * 60 + "\n")

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
            error_msg = f"[CRITICAL DATA ERROR] Expected MultiIndex, got {type(df.index)}"
            print(f"\n{error_msg}")
            raise ValueError(error_msg)
            
        # 2. Enforce Level Names and Order from Config (ADR 017 Section 1.2)
        try:
            expected_names = config["index_names"]
        except KeyError:
            raise KeyError(
                "DataFetcher Contract Violation: 'index_names' missing from config.\n"
                "To comply with ADR 017, you must explicitly define the MultiIndex levels."
            )
            
        actual_names = list(df.index.names)
        
        if actual_names[:len(expected_names)] != expected_names:
            error_msg = (
                f"[CRITICAL DATA ERROR] Index Contract Violation!\n"
                f"Expected levels: {expected_names}\n"
                f"Actual levels:   {actual_names}"
            )
            print(f"\n{error_msg}")
            raise ValueError(error_msg)
            
        # 3. Structural Normalization
        return df.reset_index()
