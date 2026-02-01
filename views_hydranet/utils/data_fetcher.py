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
