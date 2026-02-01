"""
Standalone DataFetcher component for the HydraNet pipeline.
"""

import os
from pathlib import Path

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

    def __init__(self, path_raw: str | Path) -> None:
        self.path_raw = path_raw

    def fetch(self, partition: str) -> pd.DataFrame:
        """
        Loads the DataFrame for a specific partition.

        Args:
            partition: Name of the partition (e.g., 'calibration').

        Returns:
            pd.DataFrame: The raw data as fetched from the pipeline output.
        """
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

def standardize_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes a raw VIEWS DataFrame by enforcing strict index structure.
    
    Requirements:
    - Must be a pandas MultiIndex.
    - Level 0 must be 'month_id'.
    - Level 1 must be 'priogrid_gid'.
    
    Raises:
        ValueError: If the index structure does not match exactly.
    """
    
    # 1. Enforce MultiIndex
    if not isinstance(df.index, pd.MultiIndex):
        error_msg = f"[CRITICAL DATA ERROR] Expected MultiIndex, got {type(df.index)}"
        print(f"\n{error_msg}")
        raise ValueError(error_msg)
        
    # 2. Enforce Level Names and Order
    expected_names = ["month_id", "priogrid_gid"]
    actual_names = list(df.index.names)
    
    if actual_names[:2] != expected_names:
        error_msg = (
            f"[CRITICAL DATA ERROR] Index Contract Violation!\n"
            f"Expected levels 0,1: {expected_names}\n"
            f"Actual levels:       {actual_names}"
        )
        print(f"\n{error_msg}")
        raise ValueError(error_msg)
        
    # 3. Structural Normalization
    # We bring the indices into columns so the DataSniffer and Scaler can 
    # treat them as first-class members of the DataFrame.
    return df.reset_index()
