import logging
import os
from typing import List, Dict, Any, Tuple 

from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.files.utils import read_dataframe

import pandas as pd

from views_pipeline_core.managers.model import (
    ForecastingModelManager,
    ModelPathManager,
)

from pathlib import Path

class DataLoader:

    def __init__(self, config: PipelineConfig, partition: str):
        self.config = config
        self.partition = partition
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"DataLoader initialized with config: {self.config}")

   
    def load_dataframe(self) -> pd.DataFrame:
        pass 

if __name__ == "__main__":
    



