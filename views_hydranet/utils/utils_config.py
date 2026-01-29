import logging
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

class TargetVariable(str, Enum):
    SB = "sb"
    NS = "ns"
    OS = "os"
    SB_BEST = "sb_best"
    NS_BEST = "ns_best"
    OS_BEST = "os_best"

class HydraNetConfig(BaseModel):
    """
    Strictly-typed configuration for HydraNet.
    Ensures that all required parameters are present and valid before execution.
    """
    run_type: str = Field(..., description="Partition: calibration, validation, or forecasting")
    time_steps: int = Field(..., ge=1, description="Number of months to predict")
    test_samples: int = Field(..., ge=1, description="Number of posterior samples to draw")
    input_channels: int = Field(default=3, ge=1, description="Number of input feature channels")
    target_variable: TargetVariable = Field(default=TargetVariable.SB_BEST, description="The primary target head")
    targets: List[str] = Field(default_factory=list, description="List of target column names for evaluation")
    freeze_h: str = Field(default="none", description="Memory freezing strategy")
    
    # Optional metadata
    model_time_stamp: Optional[str] = None

    @field_validator("run_type")
    @classmethod
    def validate_run_type(cls, v: str) -> str:
        valid = ["calibration", "validation", "forecasting"]
        if v not in valid:
            raise ValueError(f"run_type must be one of {valid}")
        return v

    class Config:
        extra = "allow" # Allow extra fields from the broader pipeline for now
