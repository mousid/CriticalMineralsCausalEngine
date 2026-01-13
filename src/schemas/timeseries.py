"""TimeSeries schema for canonical timeseries data."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class TimeSeriesSchema(BaseModel):
    """Canonical schema for timeseries data."""
    
    year: int = Field(..., description="Year")
    P: float = Field(..., description="Price")
    
    # Optional columns
    D: Optional[float] = Field(default=None, description="Demand")
    Q: Optional[float] = Field(default=None, description="Production/Supply")
    I: Optional[float] = Field(default=None, description="Inventory")
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = {"extra": "allow"}  # Allow additional columns beyond schema


def validate_timeseries_df(df) -> bool:
    """
    Validate a DataFrame conforms to TimeSeriesSchema.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    from pydantic import ValidationError
    import pandas as pd
    
    required_cols = ["year", "P"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Validate each row
    for idx, row in df.iterrows():
        try:
            TimeSeriesSchema(**row.to_dict())
        except ValidationError as e:
            raise ValidationError(f"Row {idx} invalid: {e}")
    
    return True

