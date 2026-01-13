"""Run report schema for scenario execution results."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path


class RunReport(BaseModel):
    """Report of a scenario run with fitted params, metrics, narrative, and artifacts."""
    
    scenario_path: str = Field(..., description="Path to scenario YAML file")
    fitted_params: Optional[Dict[str, Any]] = Field(default=None, description="Fitted parameters")
    metrics: Dict[str, float] = Field(..., description="Computed metrics")
    narrative: Optional[str] = Field(default=None, description="Narrative explanation")
    artifacts: List[str] = Field(default_factory=list, description="List of artifact file paths")

