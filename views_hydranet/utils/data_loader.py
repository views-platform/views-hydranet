import logging

import pandas as pd
from views_pipeline_core.configs.pipeline import PipelineConfig


class DataLoader:

    def __init__(self, config: PipelineConfig, partition: str):
        self.config = config
        self.partition = partition
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"DataLoader initialized with config: {self.config}")


    def load_dataframe(self) -> pd.DataFrame:
        pass

if __name__ == "__main__":
    pass




