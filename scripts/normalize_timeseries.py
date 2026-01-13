#!/usr/bin/env python3
"""Normalize messy timeseries CSV to canonical TimeSeriesSchema format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.ingest.mapping import MockLLM, infer_column_mapping, apply_mapping
from src.schemas.timeseries import TimeSeriesSchema, validate_timeseries_df


def normalize_timeseries(
    input_path: Path,
    output_path: Path,
    seed: int = 0
) -> None:
    """
    Normalize messy timeseries CSV to canonical format.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        seed: Random seed for reproducibility
    """
    # Load input data
    df = pd.read_csv(input_path)
    
    # Target columns for TimeSeriesSchema
    target_cols = ["year", "P", "D", "Q", "I"]
    
    # Infer mapping using MockLLM (deterministic)
    llm = MockLLM(seed=seed)
    mapping_result = infer_column_mapping(
        df_cols=list(df.columns),
        target_cols=target_cols,
        llm=llm,
        sample_data=df.head(5) if len(df) > 0 else None
    )
    
    # Apply mapping
    mapped_df = apply_mapping(df, mapping_result["mapping"])
    
    # Ensure year is integer, P is float
    if "year" in mapped_df.columns:
        mapped_df["year"] = pd.to_numeric(mapped_df["year"], errors="coerce").astype("Int64")
    if "P" in mapped_df.columns:
        mapped_df["P"] = pd.to_numeric(mapped_df["P"], errors="coerce").astype("float64")
    
    # Validate schema
    validate_timeseries_df(mapped_df)
    
    # Write output CSV
    mapped_df.to_csv(output_path, index=False)
    
    # Write mapping JSON
    mapping_path = output_path.with_suffix(".mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping_result, f, indent=2)
    
    # Write schema metadata JSON
    schema_path = output_path.with_suffix(".schema.json")
    schema_metadata = {
        "schema": "TimeSeriesSchema",
        "required_columns": ["year", "P"],
        "optional_columns": ["D", "Q", "I"],
        "row_count": len(mapped_df),
        "columns": list(mapped_df.columns)
    }
    with open(schema_path, "w") as f:
        json.dump(schema_metadata, f, indent=2)
    
    print(f"Normalized {len(mapped_df)} rows")
    print(f"Output: {output_path}")
    print(f"Mapping: {mapping_path}")
    print(f"Schema: {schema_path}")


def main() -> int:
    """Main entrypoint."""
    parser = argparse.ArgumentParser(description="Normalize messy timeseries CSV to canonical format")
    parser.add_argument("--in", dest="input_path", required=True, help="Input CSV file")
    parser.add_argument("--out", dest="output_path", required=True, help="Output CSV file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    normalize_timeseries(input_path, output_path, seed=args.seed)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

