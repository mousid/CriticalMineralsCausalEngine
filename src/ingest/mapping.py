"""LLM-assisted column mapping for messy input data."""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd


class BaseLLM:
    """Base class for LLM providers."""
    
    def infer_mapping(self, source_cols: List[str], target_cols: List[str], sample_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Infer mapping from source columns to target columns.
        
        Args:
            source_cols: List of source column names
            target_cols: List of target column names
            sample_data: Optional sample data for context
            
        Returns:
            Dictionary with 'mapping', 'confidence', 'rationale'
        """
        raise NotImplementedError


class MockLLM(BaseLLM):
    """Deterministic mock LLM for testing."""
    
    def __init__(self, seed: int = 0):
        """Initialize with seed for reproducibility."""
        self.seed = seed
    
    def infer_mapping(self, source_cols: List[str], target_cols: List[str], sample_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Deterministic heuristic mapping (no actual LLM call).
        
        Uses simple string matching and common aliases.
        """
        import re
        
        mapping = {}
        confidence_scores = {}
        rationale_parts = []
        
        # Common aliases mapping
        aliases = {
            "year": ["year", "yr", "y", "time", "date", "period"],
            "P": ["price", "p", "price_index", "cost", "value"],
            "D": ["demand", "d", "consumption", "usage"],
            "Q": ["supply", "q", "production", "output", "quantity"],
            "I": ["inventory", "i", "stock", "storage", "reserve"],
        }
        
        # Normalize column names for matching
        source_lower = {col.lower().strip(): col for col in source_cols}
        
        for target in target_cols:
            best_match = None
            best_confidence = 0.0
            best_rationale = ""
            
            # Check aliases
            if target in aliases:
                for alias in aliases[target]:
                    if alias in source_lower:
                        best_match = source_lower[alias]
                        best_confidence = 0.9
                        best_rationale = f"Exact alias match: '{alias}'"
                        break
            
            # If no alias match, try fuzzy matching
            if best_match is None:
                for source_col_lower, source_col in source_lower.items():
                    # Check if target is substring or vice versa
                    if target.lower() in source_col_lower or source_col_lower in target.lower():
                        confidence = 0.6
                        rationale = f"Substring match: '{source_col}' contains '{target}'"
                    # Check for common patterns
                    elif re.search(rf"\b{target.lower()}\b", source_col_lower):
                        confidence = 0.7
                        rationale = f"Word boundary match: '{source_col}'"
                    else:
                        continue
                    
                    if confidence > best_confidence:
                        best_match = source_col
                        best_confidence = confidence
                        best_rationale = rationale
            
            if best_match:
                mapping[target] = best_match
                confidence_scores[target] = best_confidence
                rationale_parts.append(f"{target} <- {best_match} ({best_rationale}, confidence={best_confidence:.2f})")
            else:
                rationale_parts.append(f"{target} <- None (no match found)")
        
        overall_confidence = sum(confidence_scores.values()) / len(target_cols) if target_cols else 0.0
        rationale = "; ".join(rationale_parts)
        
        return {
            "mapping": mapping,
            "confidence": overall_confidence,
            "rationale": rationale
        }


def infer_column_mapping(
    df_cols: List[str],
    target_cols: List[str],
    llm: BaseLLM,
    sample_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Infer column mapping from source columns to target columns using LLM.
    
    Args:
        df_cols: List of source DataFrame column names
        target_cols: List of target column names
        llm: LLM provider instance
        sample_data: Optional sample data for context
        
    Returns:
        Dictionary with 'mapping', 'confidence', 'rationale'
    """
    return llm.infer_mapping(df_cols, target_cols, sample_data)


def apply_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Apply column mapping to DataFrame.
    
    Args:
        df: Source DataFrame
        mapping: Dictionary mapping target_col -> source_col
        
    Returns:
        DataFrame with renamed columns (only mapped columns)
    """
    result = pd.DataFrame()
    
    for target_col, source_col in mapping.items():
        if source_col in df.columns:
            result[target_col] = df[source_col]
        else:
            # If source column doesn't exist, create NaN column
            result[target_col] = pd.NA
    
    return result

