import os
import pandas as pd
import numpy as np
import re
import yaml
import cloudpickle
import hashlib
import joblib
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime
from pathlib import Path
import logging
from pydantic import BaseModel, Field, validator, field_validator
import string
from functools import reduce
import warnings
from collections import Counter

from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder
)
import sys
from semiauto_classification.logger import section
logger = logging.getLogger(__name__)


class CleaningParameters(BaseModel):
    """
    Parameters for data cleaning pipeline configured using pydantic.
    All parameters have sensible defaults but can be overridden.
    """
    # Column handling
    drop_high_null_columns: bool = Field(True, description="Drop columns with high null values")
    null_column_threshold: float = Field(0.6, description="Threshold ratio for dropping high-null columns", ge=0.0,
                                         le=1.0)
    drop_constant_columns: bool = Field(True, description="Drop columns with only one unique value")
    drop_near_constant_columns: bool = Field(True, description="Drop columns with mostly one dominant value")
    dominant_value_threshold: float = Field(0.98, description="Threshold for dropping near-constant columns", ge=0.0,
                                            le=1.0)
    drop_duplicated_columns: bool = Field(True, description="Drop duplicate columns with different names")

    # Row handling
    drop_high_null_rows: bool = Field(True, description="Drop rows with too many nulls")
    null_row_threshold: float = Field(0.5, description="Threshold ratio for dropping high-null rows", ge=0.0, le=1.0)

    # Data type cleaning
    clean_strings: bool = Field(True, description="Clean string columns")
    strip_whitespace: bool = Field(True, description="Strip leading/trailing whitespace")
    lowercase_strings: bool = Field(True, description="Convert strings to lowercase")
    normalize_case: bool = Field(True, description="Normalize inconsistent case in categorical columns")
    fix_mixed_types: bool = Field(True, description="Fix columns with mixed data types")
    convert_binary_strings: bool = Field(True, description="Convert binary strings to boolean")

    # Standardize formatting
    date_format: str = Field("%Y-%m-%d", description="Standard format for dates")
    fix_dates: bool = Field(True, description="Try to fix date columns")
    standardize_formats: bool = Field(True, description="Standardize common formats (currency, percentages)")
    remove_special_chars: bool = Field(True, description="Remove special characters from strings")

    # Miscellaneous
    verbose: bool = Field(True, description="Print detailed logs")
    keep_id_columns: bool = Field(True, description="Preserve ID columns untouched")
    save_cleaning_metadata: bool = Field(True, description="Save metadata about cleaning operations")
    memory_efficient: bool = Field(True, description="Use memory-efficient operations for large datasets")

    # Processing parameters
    chunk_size: Optional[int] = Field(None, description="Process data in chunks of this size (None=no chunking)")
    max_unique_for_categorical: int = Field(100, description="Max unique values to consider a column categorical")

    # Advanced parameters
    null_filling_strategy: str = Field("none", description="Don't fill nulls - leave that to preprocessing")
    hash_threshold: int = Field(50, description="Maximum string length before converting to hash")
    correlation_threshold: float = Field(0.95, description="Threshold for highly correlated columns")

    @field_validator('null_column_threshold', 'null_row_threshold', 'dominant_value_threshold', 'correlation_threshold')
    def validate_thresholds(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {v}")
        return v


class DataCleaner:
    """
    A comprehensive data cleaning pipeline that handles:
    - High-null columns and rows
    - Constant/near-constant columns
    - String cleaning and normalization
    - Mixed types detection and conversion
    - Date standardization
    - Duplicate column detection

    This class follows the fit/transform pattern for sklearn compatibility.
    It can be fitted on training data and then applied to test data.
    """

    def __init__(self, parameters: Optional[CleaningParameters] = None):
        """Initialize the DataCleaner with optional parameters."""
        self.params = parameters if parameters is not None else CleaningParameters()
        self.stats = {
            "columns_dropped": [],
            "rows_dropped": 0,
            "fixed_columns": {},
            "original_dtypes": {},
            "new_dtypes": {},
            "start_shape": None,
            "end_shape": None,
            "cleaning_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {},
            "column_changes": {}
        }
        self.id_columns = []
        self.known_dtypes = {}
        self._is_fitted = False
        self.metadata = {}

    def __setstate__(self, state):
        """Handle potential numpy version changes during deserialization"""
        self.__dict__ = state

        # Ensure all necessary attributes exist
        if not hasattr(self, 'params'):
            self.params = CleaningParameters()
        if not hasattr(self, 'stats'):
            self.stats = {}
        if not hasattr(self, 'id_columns'):
            self.id_columns = []
        if not hasattr(self, 'known_dtypes'):
            self.known_dtypes = {}

        # Reinitialize transient attributes
        self._is_fitted = state.get('_is_fitted', False)
        self.metadata = state.get('metadata', {})

        # Handle numpy version changes
        import numpy as np
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                # Recreate array to ensure compatibility
                self.__dict__[key] = np.array(value.tolist(), dtype=value.dtype)

    def _log_column_change(self, df: pd.DataFrame, column: str, operation: str):
        """Track changes made to columns for reporting."""
        if column not in self.stats["column_changes"]:
            self.stats["column_changes"][column] = []
        self.stats["column_changes"][column].append(operation)

        # Track the actual impact if verbose
        if self.params.verbose:
            before = df[column].isna().sum()
            null_pct = round((before / len(df)) * 100, 2)
            logger.info(f"Column {column}: {operation} (contains {before} nulls, {null_pct}%)")

    def _get_intel_config(self) -> dict:
        """Load the intel.yaml configuration file - FIXED PATH"""
        try:
            current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            # FIX: Change from parent.parent to parent
            root_dir = current_dir.parent  # Now correctly points to project root
            intel_path = root_dir / 'intel.yaml'

            if not intel_path.exists():
                # Fallback remains the same
                parent_intel = root_dir.parent / 'intel.yaml'
                if parent_intel.exists():
                    intel_path = parent_intel
                else:
                    logger.warning(f"Intel config not found at {intel_path}")
                    return {}

            with open(intel_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading intel config: {str(e)}")
            return {}

    def _load_generic_feature_store(self, path: Path) -> dict:
        """Load generic feature store without dataset name"""
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading generic feature store: {str(e)}")
            return {}

    def _get_feature_store_config(self, dataset_name: str) -> dict:
        """Load feature store config - FIXED PATH"""
        try:
            current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            # FIX: Change from parent.parent to parent
            root_dir = current_dir.parent  # Now correctly points to project root
            feature_store_path = (
                    root_dir / 'references' /
                    f'feature_store_{dataset_name}' /
                    'feature_store.yaml'
            )

            if not feature_store_path.exists():
                # Fallback: try without dataset name
                generic_path = root_dir / 'references' / 'feature_store.yaml'
                if generic_path.exists():
                    logger.warning("Using generic feature_store.yaml")
                    return self._load_generic_feature_store(generic_path)

                logger.warning(f"Feature store config not found at {feature_store_path}")
                return {}

            with open(feature_store_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading feature store config: {str(e)}")
            return {}

    def _update_intel_config(self, dataset_name: str, update_dict: dict):
        """Update the intel.yaml file with cleaning information."""
        try:
            # FIX: Use consistent path calculation
            current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            root_dir = current_dir.parent  # Project root
            intel_path = root_dir / 'intel.yaml'

            if not intel_path.exists():
                logger.warning(f"Intel config not found at {intel_path}, cannot update")
                return

            with open(intel_path, 'r') as f:
                config = yaml.safe_load(f)

            # Update the config
            config.update(update_dict)

            with open(intel_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            logger.info(f"Updated intel config with cleaning information")
        except Exception as e:
            logger.error(f"Error updating intel config: {str(e)}")

    def _setup_from_config(self, dataset_name: str = None):
        """Setup the cleaner based on intel.yaml and feature_store configuration."""
        if dataset_name is None:
            intel = self._get_intel_config()
            dataset_name = intel.get('dataset_name', '')

        # Load feature store config to get column information
        feature_store = self._get_feature_store_config(dataset_name)

        # Extract ID columns that should be preserved untouched
        self.id_columns = feature_store.get('id_cols', [])
        logger.info(f"Identified ID columns to preserve: {self.id_columns}")

        # Remember original columns for filtering
        self.original_cols = feature_store.get('original_cols', [])

        # Store the dataset name
        self.dataset_name = dataset_name

        return dataset_name

    def drop_high_null_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns with high percentage of null values."""
        if not self.params.drop_high_null_columns:
            return df

        section(f"Dropping High-Null Columns (threshold: {self.params.null_column_threshold})", logger)

        # Save original shape for comparison
        original_cols = df.shape[1]

        # Calculate null percentage for each column
        null_percentages = df.isnull().mean()

        # Find columns to drop (excluding ID columns)
        high_null_cols = [
            col for col in null_percentages[null_percentages > self.params.null_column_threshold].index
            if col not in self.id_columns
        ]

        # Add to stats
        self.stats["columns_dropped"].extend([(col, "high_null") for col in high_null_cols])

        if high_null_cols:
            logger.info(
                f"Dropping {len(high_null_cols)} columns with more than {self.params.null_column_threshold * 100}% null values")
            for col in high_null_cols:
                null_pct = null_percentages[col] * 100
                logger.info(f"  - {col}: {null_pct:.2f}% null values")

            # Drop the columns
            df = df.drop(columns=high_null_cols)
            logger.info(f"Reduced columns from {original_cols} to {df.shape[1]}")
        else:
            logger.info("No high-null columns found to drop")

        return df

    def drop_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns with only one unique value (excluding nulls)."""
        if not self.params.drop_constant_columns:
            return df

        section("Dropping Constant Columns", logger)

        # Save original shape for comparison
        original_cols = df.shape[1]

        # Find constant columns (excluding ID columns)
        constant_cols = []

        for col in df.columns:
            if col in self.id_columns:
                continue

            # Get unique non-null values
            unique_values = df[col].dropna().nunique()

            if unique_values <= 1:
                constant_cols.append(col)
                self.stats["columns_dropped"].append((col, "constant"))

        if constant_cols:
            logger.info(f"Dropping {len(constant_cols)} constant columns")
            for col in constant_cols:
                try:
                    unique_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    logger.info(f"  - {col}: constant value = {unique_val}")
                except:
                    logger.info(f"  - {col}: constant column (value display error)")

            # Drop the columns
            df = df.drop(columns=constant_cols)
            logger.info(f"Reduced columns from {original_cols} to {df.shape[1]}")
        else:
            logger.info("No constant columns found to drop")

        return df

    def drop_near_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that are nearly constant (one value dominates)."""
        if not self.params.drop_near_constant_columns:
            return df

        section("Dropping Near-Constant Columns", logger)

        # Save original shape for comparison
        original_cols = df.shape[1]

        # Find near-constant columns (excluding ID columns)
        near_constant_cols = []

        for col in df.columns:
            if col in self.id_columns or col in near_constant_cols:
                continue

            # Skip if too many nulls to determine
            if df[col].isna().mean() > 0.5:
                continue

            # Get value counts
            value_counts = df[col].value_counts(normalize=True, dropna=True)

            # Check if the most common value exceeds the threshold
            if not value_counts.empty and value_counts.iloc[0] > self.params.dominant_value_threshold:
                near_constant_cols.append(col)
                self.stats["columns_dropped"].append((col, "near_constant"))

        if near_constant_cols:
            logger.info(
                f"Dropping {len(near_constant_cols)} near-constant columns (dominant value > {self.params.dominant_value_threshold * 100}%)")
            for col in near_constant_cols:
                try:
                    value_counts = df[col].value_counts(normalize=True, dropna=True)
                    dominant_val = value_counts.index[0]
                    dominant_pct = value_counts.iloc[0] * 100
                    logger.info(f"  - {col}: dominant value '{dominant_val}' appears in {dominant_pct:.2f}% of records")
                except:
                    logger.info(f"  - {col}: near-constant column (value display error)")

            # Drop the columns
            df = df.drop(columns=near_constant_cols)
            logger.info(f"Reduced columns from {original_cols} to {df.shape[1]}")
        else:
            logger.info("No near-constant columns found to drop")

        return df

    def drop_duplicated_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that are duplicates of other columns but with different names."""
        if not self.params.drop_duplicated_columns:
            return df

        section("Dropping Duplicated Columns", logger)

        # Save original shape for comparison
        original_cols = df.shape[1]

        # Dictionary to store duplicate columns
        duplicates = {}

        # Find duplicate columns
        for i, col1 in enumerate(df.columns):
            # Skip ID columns and already identified duplicates
            if col1 in self.id_columns or any(col1 in dupes for dupes in duplicates.values()):
                continue

            duplicates[col1] = []

            # Compare with other columns
            for col2 in df.columns[i + 1:]:
                if col2 in self.id_columns:
                    continue

                # Check if columns are identical
                if df[col1].equals(df[col2]):
                    duplicates[col1].append(col2)
                    self.stats["columns_dropped"].append((col2, f"duplicate_of_{col1}"))

        # Remove empty entries
        duplicates = {k: v for k, v in duplicates.items() if v}

        if duplicates:
            logger.info(f"Found {sum(len(v) for v in duplicates.values())} duplicate columns")

            # List all duplicates
            for original, dupes in duplicates.items():
                logger.info(f"  - {original} has duplicates: {', '.join(dupes)}")

            # Create list of all duplicates to drop
            to_drop = [dup for dupes in duplicates.values() for dup in dupes]

            # Drop the columns
            df = df.drop(columns=to_drop)
            logger.info(f"Reduced columns from {original_cols} to {df.shape[1]}")
        else:
            logger.info("No duplicate columns found")

        return df

    def clean_string_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean string columns: lowercase, strip whitespace, remove special characters."""
        if not self.params.clean_strings:
            return df

        section("Cleaning String Columns", logger)

        # Process object/string columns
        string_cols = df.select_dtypes(include=['object']).columns

        # Filter out ID columns
        string_cols = [col for col in string_cols if col not in self.id_columns]

        if len(string_cols) > 0:
            logger.info(f"Cleaning {len(string_cols)} string columns")

            for col in string_cols:
                # Skip columns with high null ratios
                if df[col].isna().mean() > 0.5:
                    logger.info(f"  - Skipping {col} (>50% nulls)")
                    continue

                # Make a copy to avoid chained assignment warning
                df_col = df[col].copy()

                # Only process string data
                string_mask = df_col.apply(lambda x: isinstance(x, str))
                if string_mask.sum() == 0:
                    continue

                # Track if column was modified
                modified = False

                # Strip whitespace
                if self.params.strip_whitespace:
                    original_values = df_col[string_mask].copy()
                    df_col[string_mask] = df_col[string_mask].str.strip()
                    if not df_col[string_mask].equals(original_values):
                        modified = True
                        self._log_column_change(df, col, "stripped_whitespace")

                # Convert to lowercase
                if self.params.lowercase_strings:
                    original_values = df_col[string_mask].copy()
                    df_col[string_mask] = df_col[string_mask].str.lower()
                    if not df_col[string_mask].equals(original_values):
                        modified = True
                        self._log_column_change(df, col, "lowercased")

                # Remove special characters
                if self.params.remove_special_chars:
                    original_values = df_col[string_mask].copy()
                    # Only keep alphanumeric and basic punctuation
                    df_col[string_mask] = df_col[string_mask].apply(
                        lambda x: re.sub(r'[^\w\s.,;:!?()-]', '', x) if isinstance(x, str) else x
                    )
                    if not df_col[string_mask].equals(original_values):
                        modified = True
                        self._log_column_change(df, col, "removed_special_chars")

                # Update column in dataframe if modified
                if modified:
                    self.stats["fixed_columns"][col] = "string_cleaning"
                    df[col] = df_col
                    logger.info(f"  - Cleaned string column: {col}")
        else:
            logger.info("No string columns found to clean")

        return df

    def fix_mixed_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix columns with mixed data types by converting to the most appropriate type.
        This handles common cases like numbers stored as strings and boolean values.
        """
        if not self.params.fix_mixed_types:
            return df

        section("Fixing Mixed Types", logger)

        # Only process non-ID columns
        cols_to_process = [col for col in df.columns if col not in self.id_columns]

        # Store original dtypes for reference
        self.stats["original_dtypes"] = {col: str(df[col].dtype) for col in df.columns}

        # Process each column
        for col in cols_to_process:
            # Skip if the column has too many nulls
            if df[col].isna().mean() > 0.5:
                continue

            # Get current dtype
            current_dtype = df[col].dtype

            # Only process object columns
            if current_dtype != 'object':
                continue

            # Make a copy of the column to avoid SettingWithCopyWarning
            series = df[col].copy()

            # Skip columns with long string values
            if series.dropna().astype(str).str.len().max() > 100:
                continue

            # Try converting to numeric
            numeric_series = pd.to_numeric(series, errors='coerce')
            numeric_conversion_success = numeric_series.notna().mean() > 0.8  # 80% success rate

            # Try converting to datetime
            if not numeric_conversion_success and self.params.fix_dates:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        datetime_series = pd.to_datetime(series, errors='coerce')
                    datetime_conversion_success = datetime_series.notna().mean() > 0.8  # 80% success rate

                    if datetime_conversion_success:
                        df[col] = datetime_series
                        logger.info(f"Converted {col} to datetime")
                        self._log_column_change(df, col, "converted_to_datetime")
                        self.stats["fixed_columns"][col] = "datetime_conversion"
                        continue
                except:
                    pass

            # Try converting to numeric if high success rate
            if numeric_conversion_success:
                # Check if values are mostly integers
                if pd.notna(numeric_series).all() and np.floor(numeric_series) == numeric_series:
                    df[col] = numeric_series.astype('Int64')  # Int64 allows NaN values
                    logger.info(f"Converted {col} to integer")
                    self._log_column_change(df, col, "converted_to_integer")
                else:
                    df[col] = numeric_series
                    logger.info(f"Converted {col} to float")
                    self._log_column_change(df, col, "converted_to_float")

                self.stats["fixed_columns"][col] = "numeric_conversion"
                continue

            # Check for boolean values
            if self.params.convert_binary_strings:
                # Define common boolean mappings
                true_values = ['yes', 'y', 'true', 't', '1', 'on']
                false_values = ['no', 'n', 'false', 'f', '0', 'off']

                # Convert to lowercase for comparison
                lowercase_series = series.astype(str).str.lower()

                # Check if values match boolean patterns
                is_bool = lowercase_series.isin(true_values + false_values).all()

                if is_bool:
                    # Convert to boolean
                    df[col] = lowercase_series.isin(true_values)
                    logger.info(f"Converted {col} to boolean")
                    self._log_column_change(df, col, "converted_to_boolean")
                    self.stats["fixed_columns"][col] = "boolean_conversion"

        # Store new dtypes
        self.stats["new_dtypes"] = {col: str(df[col].dtype) for col in df.columns}

        return df

    def remove_high_null_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with too many null values."""
        if not self.params.drop_high_null_rows:
            return df

        section(f"Removing High-Null Rows (threshold: {self.params.null_row_threshold})", logger)

        # Save original shape for comparison
        original_rows = df.shape[0]

        # Calculate null ratio for each row
        null_ratio = df.isnull().sum(axis=1) / df.shape[1]

        # Find rows to drop
        rows_to_drop = null_ratio[null_ratio > self.params.null_row_threshold].index

        if len(rows_to_drop) > 0:
            # Calculate percentage of rows to drop
            drop_percentage = (len(rows_to_drop) / original_rows) * 100

            # Warn if dropping too many rows
            if drop_percentage > 20:
                logger.warning(f"Removing {drop_percentage:.2f}% of rows due to high null values!")

            # Drop the rows
            df = df.drop(index=rows_to_drop)

            # Update stats
            self.stats["rows_dropped"] += len(rows_to_drop)

            logger.info(
                f"Removed {len(rows_to_drop)} rows with more than {self.params.null_row_threshold * 100}% null values")
            logger.info(f"Reduced rows from {original_rows} to {df.shape[0]}")
        else:
            logger.info("No high-null rows found to remove")

        return df

    def normalize_case_in_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize inconsistent case in categorical columns."""
        if not self.params.normalize_case:
            return df

        section("Normalizing Case in Categorical Columns", logger)

        # Process string columns that might be categorical
        obj_cols = df.select_dtypes(include=['object']).columns

        # Filter out ID columns and columns with very high cardinality
        categorical_cols = []
        for col in obj_cols:
            if col in self.id_columns:
                continue

            # Check if column is likely categorical (limited unique values)
            unique_count = df[col].nunique()
            if unique_count <= self.params.max_unique_for_categorical and unique_count > 1:
                categorical_cols.append(col)

        if categorical_cols:
            logger.info(f"Checking {len(categorical_cols)} potential categorical columns for case inconsistencies")

            for col in categorical_cols:
                # Skip if too many nulls
                if df[col].isna().mean() > 0.5:
                    continue

                # Get string values only
                string_values = df[col][df[col].apply(lambda x: isinstance(x, str))]

                if len(string_values) == 0:
                    continue

                # Check for case inconsistencies by comparing lowercase versions
                lowercase_values = string_values.str.lower()
                case_inconsistent = False

                # Group by lowercase and check if same value appears with different cases
                case_groups = {}
                for original, lower in zip(string_values, lowercase_values):
                    if lower not in case_groups:
                        case_groups[lower] = []
                    if original not in case_groups[lower]:
                        case_groups[lower].append(original)

                # Find groups with multiple case variants
                inconsistent_groups = {k: v for k, v in case_groups.items() if len(v) > 1}

                if inconsistent_groups:
                    case_inconsistent = True
                    logger.info(f"  - Found case inconsistencies in {col}:")
                    for lower, variants in inconsistent_groups.items():
                        logger.info(f"    * '{lower}' appears as: {', '.join(variants)}")

                    # Create mapping from inconsistent cases to the most common form
                    mapping = {}
                    for lower, variants in inconsistent_groups.items():
                        # Find most common variant
                        variant_counts = Counter([v for v in string_values if v.lower() == lower])
                        most_common = variant_counts.most_common(1)[0][0]

                        # Map all variants to the most common
                        for variant in variants:
                            if variant != most_common:
                                mapping[variant] = most_common

                    # Apply mapping
                    df[col] = df[col].replace(mapping)
                    logger.info(f"  - Normalized case inconsistencies in {col}")
                    self._log_column_change(df, col, "normalized_case")
                    self.stats["fixed_columns"][col] = "case_normalization"
        else:
            logger.info("No suitable categorical columns found for case normalization")

        return df

    def standardize_date_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize date/datetime columns to a consistent format."""
        if not self.params.fix_dates:
            return df

        section("Standardizing Date Formats", logger)

        # Only process non-ID columns
        cols_to_process = [col for col in df.columns if col not in self.id_columns]

        # Process each column
        for col in cols_to_process:
            # Skip if already datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # Format datetime columns consistently
                df[col] = pd.to_datetime(df[col]).dt.strftime(self.params.date_format)
                self._log_column_change(df, col, "standardized_date_format")
                self.stats["fixed_columns"][col] = "date_standardization"
                continue

            # Skip non-string columns
            if df[col].dtype != 'object':
                continue

            # Check for date patterns in the column
            sample_values = df[col].dropna().astype(str).sample(min(100, len(df[col].dropna())))
            date_patterns = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y', '%Y%m%d']

            # Try to convert sample to datetime using different patterns
            for pattern in date_patterns:
                try:
                    converted = pd.to_datetime(sample_values, format=pattern, errors='coerce')
                    if converted.notna().mean() > 0.8:  # 80% success rate
                        # Apply to full column
                        df[col] = pd.to_datetime(df[col], format=pattern, errors='coerce')
                        df[col] = df[col].dt.strftime(self.params.date_format)
                        logger.info(f"Standardized date format in {col} using pattern {pattern}")
                        self._log_column_change(df, col, f"standardized_date_format_{pattern}")
                        self.stats["fixed_columns"][col] = "date_standardization"
                        break
                except:
                    continue

        return df

    def basic_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps in sequence."""
        section("Starting Basic Data Cleaning", logger)

        # Store original shape
        self.stats["start_shape"] = df.shape

        # Apply cleaning steps in logical order
        df = self.drop_high_null_columns(df)
        df = self.drop_constant_columns(df)
        df = self.drop_near_constant_columns(df)
        df = self.drop_duplicated_columns(df)
        df = self.clean_string_columns(df)
        df = self.normalize_case_in_categorical(df)
        df = self.fix_mixed_types(df)
        df = self.standardize_date_formats(df)
        df = self.remove_high_null_rows(df)

        # Store final shape
        self.stats["end_shape"] = df.shape

        # Calculate summary statistics
        self.stats["summary"] = {
            "columns_dropped": len(self.stats["columns_dropped"]),
            "rows_dropped": self.stats["rows_dropped"],
            "columns_fixed": len(self.stats["fixed_columns"]),
            "start_rows": self.stats["start_shape"][0],
            "start_cols": self.stats["start_shape"][1],
            "end_rows": self.stats["end_shape"][0],
            "end_cols": self.stats["end_shape"][1],
        }

        # Log summary
        section("Data Cleaning Summary", logger)
        logger.info(f"Starting shape: {self.stats['start_shape'][0]} rows × {self.stats['start_shape'][1]} columns")
        logger.info(f"Ending shape: {self.stats['end_shape'][0]} rows × {self.stats['end_shape'][1]} columns")
        logger.info(f"Columns dropped: {self.stats['summary']['columns_dropped']}")
        logger.info(f"Rows dropped: {self.stats['summary']['rows_dropped']}")
        logger.info(f"Columns fixed/modified: {self.stats['summary']['columns_fixed']}")

        return df

    def fit(self, df: pd.DataFrame, dataset_name: str = None) -> 'DataCleaner':
        """
        Fit the cleaning pipeline on training data.
        This captures the state needed to apply the same transformations to test data.

        Args:
            df: Pandas DataFrame with training data
            dataset_name: Optional name of the dataset used to load config

        Returns:
            self: The fitted DataCleaner instance
        """
        section("Fitting Data Cleaning Pipeline", logger)

        # Setup from configuration if dataset name provided
        if dataset_name:
            self._setup_from_config(dataset_name)

        # Store the column information pre-cleaning
        self.original_cols = df.columns.tolist()
        self.original_shape = df.shape

        # Store dtypes for later reference
        self.known_dtypes = {col: df[col].dtype for col in df.columns}

        # Apply the cleaning pipeline
        cleaned_df = self.basic_data_cleaning(df.copy())

        # Save the column state after cleaning
        self.cleaned_cols = cleaned_df.columns.tolist()

        # Store correlation information for high-correlation columns
        if self.params.correlation_threshold < 1.0:
            self._store_correlation_info(cleaned_df)

        # Mark as fitted
        self._is_fitted = True

        # Return self for chaining
        return self

    def _store_correlation_info(self, df: pd.DataFrame) -> None:
        """Store information about highly correlated columns."""
        # Only calculate for numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # Skip if not enough numeric columns
        if len(numeric_cols) < 2:
            logger.info("Not enough numeric columns to calculate correlations")
            return

        # Calculate correlation matrix
        try:
            corr_matrix = df[numeric_cols].corr().abs()

            # Create pairs of highly correlated columns
            self.correlated_cols = []

            # Get upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            # Find column pairs with correlation > threshold
            high_corr_pairs = [(upper.index[i], upper.columns[j], upper.iloc[i, j])
                               for i, j in zip(*np.where(upper > self.params.correlation_threshold))]

            if high_corr_pairs:
                logger.info(
                    f"Found {len(high_corr_pairs)} pairs of highly correlated columns (r > {self.params.correlation_threshold})")
                for col1, col2, corr in high_corr_pairs:
                    logger.info(f"  - {col1} and {col2}: r = {corr:.4f}")
                    self.correlated_cols.append((col1, col2, corr))
            else:
                logger.info(f"No highly correlated columns found (threshold: {self.params.correlation_threshold})")

        except Exception as e:
            logger.warning(f"Failed to calculate correlations: {str(e)}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the fitted cleaning pipeline.

        Args:
            df: Pandas DataFrame to clean

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if not self._is_fitted:
            raise ValueError("DataCleaner must be fitted before transform can be called")

        section("Transforming Data with Fitted Cleaning Pipeline", logger)

        # Apply the cleaning steps
        return self.basic_data_cleaning(df.copy())

    def fit_transform(self, df: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:
        """
        Convenience method to fit and transform in one step.

        Args:
            df: Pandas DataFrame to clean
            dataset_name: Optional name of the dataset used to load config

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        return self.fit(df, dataset_name).transform(df)

    def save(self, path: str) -> None:
        """
        Save the fitted DataCleaner to disk using cloudpickle.

        Args:
            path: Path where to save the pipeline
        """
        if not self._is_fitted:
            raise ValueError("Cannot save an unfitted DataCleaner")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save using cloudpickle for better serialization
            with open(path, 'wb') as f:
                cloudpickle.dump(self, f)

            logger.info(f"Saved data cleaning pipeline to {path}")
        except Exception as e:
            logger.error(f"Failed to save pipeline: {str(e)}")
            raise

    @classmethod
    def load(cls, path: str) -> 'DataCleaner':
        """
        Load a fitted DataCleaner from disk.

        Args:
            path: Path from where to load the pipeline

        Returns:
            DataCleaner: Loaded pipeline
        """
        try:
            with open(path, 'rb') as f:
                pipeline = cloudpickle.load(f)

            logger.info(f"Loaded data cleaning pipeline from {path}")
            return pipeline
        except Exception as e:
            logger.error(f"Failed to load pipeline: {str(e)}")
            raise

    def get_statistics(self) -> dict:
        """Get summary statistics of the cleaning process."""
        if not self._is_fitted:
            raise ValueError("DataCleaner must be fitted before statistics can be retrieved")

        return self.stats

    def save_cleaning_report(self, path: str) -> None:
        """
        Save a detailed cleaning report to a file.

        Args:
            path: Path where to save the report
        """
        if not self._is_fitted:
            raise ValueError("DataCleaner must be fitted before generating a report")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Create report as markdown
            with open(path, 'w') as f:
                f.write("# Data Cleaning Report\n\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("## Summary\n\n")
                f.write(
                    f"* Starting shape: {self.stats['start_shape'][0]} rows × {self.stats['start_shape'][1]} columns\n")
                f.write(f"* Ending shape: {self.stats['end_shape'][0]} rows × {self.stats['end_shape'][1]} columns\n")
                f.write(f"* Columns dropped: {self.stats['summary']['columns_dropped']}\n")
                f.write(f"* Rows dropped: {self.stats['summary']['rows_dropped']}\n")
                f.write(f"* Columns fixed/modified: {self.stats['summary']['columns_fixed']}\n\n")

                f.write("## Columns Dropped\n\n")
                if self.stats["columns_dropped"]:
                    f.write("| Column | Reason |\n")
                    f.write("|--------|--------|\n")
                    for col, reason in self.stats["columns_dropped"]:
                        f.write(f"| {col} | {reason} |\n")
                else:
                    f.write("No columns were dropped.\n")

                f.write("\n## Columns Modified\n\n")
                if self.stats["fixed_columns"]:
                    f.write("| Column | Modification |\n")
                    f.write("|--------|-------------|\n")
                    for col, mod in self.stats["fixed_columns"].items():
                        f.write(f"| {col} | {mod} |\n")
                else:
                    f.write("No columns were modified.\n")

                f.write("\n## Data Type Changes\n\n")
                f.write("| Column | Original Type | New Type |\n")
                f.write("|--------|--------------|----------|\n")

                type_changes = {}
                for col in self.stats["original_dtypes"]:
                    if col in self.stats["new_dtypes"] and self.stats["original_dtypes"][col] != \
                            self.stats["new_dtypes"][col]:
                        type_changes[col] = (self.stats["original_dtypes"][col], self.stats["new_dtypes"][col])

                if type_changes:
                    for col, (orig_type, new_type) in type_changes.items():
                        f.write(f"| {col} | {orig_type} | {new_type} |\n")
                else:
                    f.write("| No data type changes were made | | |\n")

                # Add correlation information if available
                if hasattr(self, 'correlated_cols') and self.correlated_cols:
                    f.write("\n## Highly Correlated Columns\n\n")
                    f.write("| Column 1 | Column 2 | Correlation |\n")
                    f.write("|----------|----------|-------------|\n")
                    for col1, col2, corr in self.correlated_cols:
                        f.write(f"| {col1} | {col2} | {corr:.4f} |\n")

            logger.info(f"Saved cleaning report to {path}")
        except Exception as e:
            logger.error(f"Failed to save cleaning report: {str(e)}")
            raise


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for handling outliers using either IQR or Z-Score method.
    """

    def __init__(self, method: str = 'IQR', columns: List[str] = None):
        """
        Initialize the OutlierHandler.

        Args:
            method (str): Method to use for outlier detection ('IQR' or 'Z-Score')
            columns (List[str]): List of columns to handle outliers for
        """
        self.method = method
        self.columns = columns
        self.thresholds = {}

        # Validate method
        if method not in ['IQR', 'Z-Score']:
            raise ValueError("Method must be either 'IQR' or 'Z-Score'")

        logger.info(f"Initialized OutlierHandler with method: {method}")

    def fit(self, X, y=None):
        """
        Calculate the thresholds for outlier detection.

        Args:
            X (pd.DataFrame): Input data
            y: Ignored

        Returns:
            self
        """
        # Only process if columns are provided
        if not self.columns:
            logger.warning("No columns provided for outlier handling")
            return self

        # Calculate thresholds for each column
        for col in self.columns:
            if col not in X.columns:
                logger.warning(f"Column {col} not found in input data")
                continue

            if self.method == 'IQR':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                self.thresholds[col] = {'lower': lower_bound, 'upper': upper_bound}
                logger.info(f"IQR thresholds for {col}: lower={lower_bound:.4f}, upper={upper_bound:.4f}")

            elif self.method == 'Z-Score':
                mean = X[col].mean()
                std = X[col].std()

                self.thresholds[col] = {'mean': mean, 'std': std}
                logger.info(f"Z-Score parameters for {col}: mean={mean:.4f}, std={std:.4f}")

        return self

    def transform(self, X):
        """
        Handle outliers in the data according to the fitted thresholds.

        Args:
            X (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Transformed data with outliers handled
        """
        X_transformed = X.copy()

        if not self.columns or not self.thresholds:
            logger.warning("No columns or thresholds available for outlier handling")
            return X_transformed

        for col in self.columns:
            if col not in X_transformed.columns or col not in self.thresholds:
                continue

            if self.method == 'IQR':
                lower = self.thresholds[col]['lower']
                upper = self.thresholds[col]['upper']

                # Cap outliers at thresholds
                outliers_lower = X_transformed[col] < lower
                outliers_upper = X_transformed[col] > upper

                if outliers_lower.any() or outliers_upper.any():
                    count_lower = outliers_lower.sum()
                    count_upper = outliers_upper.sum()
                    logger.info(
                        f"Handling outliers in {col}: {count_lower} below lower bound, {count_upper} above upper bound")

                    X_transformed.loc[outliers_lower, col] = lower
                    X_transformed.loc[outliers_upper, col] = upper

            elif self.method == 'Z-Score':
                mean = self.thresholds[col]['mean']
                std = self.thresholds[col]['std']

                # Identify outliers (Z-Score > 3 or < -3)
                z_scores = (X_transformed[col] - mean) / std
                outliers = (z_scores > 3) | (z_scores < -3)

                if outliers.any():
                    count = outliers.sum()
                    logger.info(f"Handling {count} Z-Score outliers in {col}")

                    # Cap outliers at ±3 standard deviations
                    X_transformed.loc[z_scores > 3, col] = mean + 3 * std
                    X_transformed.loc[z_scores < -3, col] = mean - 3 * std

        return X_transformed


class IDColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, id_cols: List[str]):
        self.id_columns = id_cols
        logger.info(f"Initialized ID column dropper with columns: {id_cols}")

    def fit(self, X, y=None):
        # Verify which ID columns exist in the dataframe
        existing_cols = [col for col in self.id_columns if col in X.columns]
        logger.info(f"Found {len(existing_cols)}/{len(self.id_columns)} ID columns in data")
        return self

    def transform(self, X):
        columns_to_drop = [col for col in self.id_columns if col in X.columns]
        if columns_to_drop:
            logger.info(f"Dropping ID columns: {columns_to_drop}")
            return X.drop(columns=columns_to_drop)
        return X


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for handling missing values using specified method.
    """

    def __init__(self, method: str = 'mean', columns: List[str] = None):
        """
        Initialize the MissingValueHandler.

        Args:
            method (str): Method to use for handling missing values ('mean', 'median', 'mode', 'drop')
            columns (List[str]): List of columns to handle missing values for
        """
        self.method = method
        self.columns = columns
        self.fill_values = {}

        # Validate method
        if method not in ['mean', 'median', 'mode', 'drop']:
            raise ValueError("Method must be one of 'mean', 'median', 'mode', or 'drop'")

        logger.info(f"Initialized MissingValueHandler with method: {method}")

    def fit(self, X, y=None):
        """
        Calculate the fill values for missing value handling.

        Args:
            X (pd.DataFrame): Input data
            y: Ignored

        Returns:
            self
        """
        # Handle drop method separately (nothing to fit)
        if self.method == 'drop':
            return self

        # Only process if columns are provided
        if not self.columns:
            logger.warning("No columns provided for missing value handling")
            return self

        # Calculate fill values for each column
        for col in self.columns:
            if col not in X.columns:
                logger.warning(f"Column {col} not found in input data")
                continue

            # Check if the method is appropriate for the column's data type
            if self.method in ['mean', 'median']:
                if pd.api.types.is_numeric_dtype(X[col]):
                    if self.method == 'mean':
                        self.fill_values[col] = X[col].mean()
                    elif self.method == 'median':
                        self.fill_values[col] = X[col].median()
                    logger.info(f"{self.method.capitalize()} value for {col}: {self.fill_values[col]:.4f}")
                else:
                    # Fallback to mode for non-numeric columns if mean/median is selected
                    logger.warning(f"Column {col} is non-numeric. Using mode instead of {self.method}.")
                    self.fill_values[col] = X[col].mode()[0]
                    logger.info(f"Mode value for {col}: {self.fill_values[col]}")
            elif self.method == 'mode':
                self.fill_values[col] = X[col].mode()[0]
                logger.info(f"Mode value for {col}: {self.fill_values[col]}")

        return self

    def transform(self, X):
        """
        Handle missing values in the data according to the fitted values.

        Args:
            X (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Transformed data with missing values handled
        """
        X_transformed = X.copy()

        if self.method == 'drop':
            # Drop rows with missing values
            rows_before = len(X_transformed)
            X_transformed = X_transformed.dropna(subset=self.columns)
            rows_after = len(X_transformed)
            rows_dropped = rows_before - rows_after

            if rows_dropped > 0:
                logger.info(f"Dropped {rows_dropped} rows with missing values")

            return X_transformed

        if not self.columns or not self.fill_values:
            logger.warning("No columns or fill values available for missing value handling")
            return X_transformed

        for col in self.columns:
            if col not in X_transformed.columns or col not in self.fill_values:
                continue

            # Count missing values before filling
            missing_count = X_transformed[col].isna().sum()

            if missing_count > 0:
                # Fill missing values
                X_transformed[col] = X_transformed[col].fillna(self.fill_values[col])
                logger.info(f"Filled {missing_count} missing values in {col} with {self.fill_values[col]}")

        return X_transformed


class SkewedDataHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for handling skewed data using power transformers.
    """

    def __init__(self, method: str = 'yeo-johnson', columns: List[str] = None):
        """
        Initialize the SkewedDataHandler.

        Args:
            method (str): Method to use for handling skewed data ('yeo-johnson' or 'box-cox')
            columns (List[str]): List of columns to handle skewed data for
        """
        self.method = method
        self.columns = columns
        self.transformers = {}

        # Validate method
        if method not in ['yeo-johnson', 'box-cox']:
            raise ValueError("Method must be either 'yeo-johnson' or 'box-cox'")

        logger.info(f"Initialized SkewedDataHandler with method: {method}")

    def fit(self, X, y=None):
        """
        Fit power transformers for skewed data handling.

        Args:
            X (pd.DataFrame): Input data
            y: Ignored

        Returns:
            self
        """
        # Only process if columns are provided
        if not self.columns:
            logger.warning("No columns provided for skewed data handling")
            return self

        # Fit transformer for each column
        for col in self.columns:
            if col not in X.columns:
                logger.warning(f"Column {col} not found in input data")
                continue

            # Create and fit transformer
            transformer = PowerTransformer(method=self.method, standardize=True)

            # Box-Cox requires positive values
            if self.method == 'box-cox' and X[col].min() <= 0:
                logger.warning(f"Column {col} has values <= 0, which is incompatible with Box-Cox transform.")
                logger.warning(f"Shifting data to positive values for Box-Cox transformation")
                shift_value = abs(X[col].min()) + 1.0  # Add 1 to ensure all values are positive
                self.transformers[col] = {'transformer': transformer, 'shift': shift_value}
                # Fit transformer on shifted data
                transformer.fit(X[col].add(shift_value).values.reshape(-1, 1))
            else:
                self.transformers[col] = {'transformer': transformer, 'shift': 0.0}
                # Fit transformer
                transformer.fit(X[col].values.reshape(-1, 1))

            logger.info(f"Fitted power transformer for {col}")

        return self

    def transform(self, X):
        """
        Transform skewed data using fitted power transformers.

        Args:
            X (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Transformed data with skewed data handled
        """
        X_transformed = X.copy()

        if not self.columns or not self.transformers:
            logger.warning("No columns or transformers available for skewed data handling")
            return X_transformed

        for col in self.columns:
            if col not in X_transformed.columns or col not in self.transformers:
                continue

            # Get transformer and shift value
            transformer = self.transformers[col]['transformer']
            shift_value = self.transformers[col]['shift']

            # Apply transformation
            if shift_value > 0:
                # Apply shift before transformation
                transformed_values = transformer.transform(X_transformed[col].add(shift_value).values.reshape(-1, 1))
            else:
                transformed_values = transformer.transform(X_transformed[col].values.reshape(-1, 1))

            # Replace original values with transformed values
            X_transformed[col] = transformed_values.flatten()
            logger.info(f"Transformed skewed data in {col}")

        return X_transformed


class NumericalScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for scaling numerical features.
    """

    def __init__(self, method: str = 'standard', columns: List[str] = None):
        """
        Initialize the NumericalScaler.

        Args:
            method (str): Method to use for scaling ('standard', 'robust', 'minmax')
            columns (List[str]): List of columns to scale
        """
        self.method = method
        self.columns = columns
        self.scalers = {}

        # Validate method
        if method not in ['standard', 'robust', 'minmax']:
            raise ValueError("Method must be one of 'standard', 'robust', or 'minmax'")

        logger.info(f"Initialized NumericalScaler with method: {method}")

    def fit(self, X, y=None):
        """
        Fit scalers for numerical features.

        Args:
            X (pd.DataFrame): Input data
            y: Ignored

        Returns:
            self
        """
        # Only process if columns are provided
        if not self.columns:
            logger.warning("No columns provided for scaling")
            return self

        # Initialize and fit scaler for each column
        for col in self.columns:
            if col not in X.columns:
                logger.warning(f"Column {col} not found in input data")
                continue

            # Create and fit appropriate scaler
            if self.method == 'standard':
                scaler = StandardScaler()
            elif self.method == 'robust':
                scaler = RobustScaler()
            elif self.method == 'minmax':
                scaler = MinMaxScaler()

            # Fit scaler
            scaler.fit(X[col].values.reshape(-1, 1))
            self.scalers[col] = scaler
            logger.info(f"Fitted {self.method} scaler for {col}")

        return self

    def transform(self, X):
        """
        Scale numerical features using fitted scalers.

        Args:
            X (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Transformed data with scaled numerical features
        """
        X_transformed = X.copy()

        if not self.columns or not self.scalers:
            logger.warning("No columns or scalers available for scaling")
            return X_transformed

        for col in self.columns:
            if col not in X_transformed.columns or col not in self.scalers:
                continue

            # Get scaler for this column
            scaler = self.scalers[col]

            # Apply scaling
            scaled_values = scaler.transform(X_transformed[col].values.reshape(-1, 1))
            X_transformed[col] = scaled_values.flatten()
            logger.info(f"Scaled numerical features in {col}")

        return X_transformed


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Custom transformer for encoding categorical features.
    """

    def __init__(self, method: str = 'onehot', columns: List[str] = None, drop_first: bool = True):
        """
        Initialize the CategoricalEncoder.

        Args:
            method (str): Method to use for encoding ('onehot', 'label', 'dummies')
            columns (List[str]): List of columns to encode
            drop_first (bool): Whether to drop the first category in one-hot encoding
        """
        self.method = method
        self.columns = columns
        self.drop_first = drop_first
        self.encoders = {}
        self.dummy_columns = {}

        # Validate method
        if method not in ['onehot', 'label', 'dummies']:
            raise ValueError("Method must be one of 'onehot', 'label', or 'dummies'")

        logger.info(f"Initialized CategoricalEncoder with method: {method}")

    def fit(self, X, y=None):
        """
        Fit encoders for categorical features.

        Args:
            X (pd.DataFrame): Input data
            y: Ignored

        Returns:
            self
        """
        # Only process if columns are provided
        if not self.columns:
            logger.warning("No columns provided for categorical encoding")
            return self

        # Initialize and fit encoder for each column
        for col in self.columns:
            if col not in X.columns:
                logger.warning(f"Column {col} not found in input data")
                continue

            if self.method == 'onehot':
                encoder = OneHotEncoder(
                    sparse_output=False,
                    drop='first' if self.drop_first else None,
                    handle_unknown='ignore'  # Add this parameter
                )
                encoder.fit(X[col].values.reshape(-1, 1))
                self.encoders[col] = encoder
                logger.info(f"Fitted OneHotEncoder for {col}")

                # Store feature names for later
                feature_names = encoder.get_feature_names_out([col])
                self.dummy_columns[col] = feature_names.tolist()

            elif self.method == 'label':
                encoder = LabelEncoder()
                encoder.fit(X[col])
                self.encoders[col] = encoder
                logger.info(f"Fitted LabelEncoder for {col}")

            elif self.method == 'dummies':
                # For 'dummies' method, we'll just store unique values
                # actual transformation will be done in transform method
                unique_values = X[col].unique()
                self.encoders[col] = unique_values

                # Create dummy column names (will be used during transform)
                dummy_cols = [f"{col}_{val}" for val in unique_values]
                if self.drop_first:
                    dummy_cols = dummy_cols[1:]
                self.dummy_columns[col] = dummy_cols

                logger.info(f"Stored unique values for pd.get_dummies for {col}")

        return self

    def transform(self, X):
        """
        Encode categorical features using fitted encoders.

        Args:
            X (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Transformed data with encoded categorical features
        """
        X_transformed = X.copy()

        if not self.columns or not self.encoders:
            logger.warning("No columns or encoders available for categorical encoding")
            return X_transformed

        for col in self.columns:
            if col not in X_transformed.columns or col not in self.encoders:
                continue

            if self.method == 'onehot':
                # Apply one-hot encoding
                encoder = self.encoders[col]
                encoded_array = encoder.transform(X_transformed[col].values.reshape(-1, 1))

                # Create DataFrame with encoded values
                encoded_df = pd.DataFrame(
                    encoded_array,
                    columns=self.dummy_columns[col],
                    index=X_transformed.index
                )

                # Drop original column and join encoded columns
                X_transformed = X_transformed.drop(columns=[col])
                X_transformed = pd.concat([X_transformed, encoded_df], axis=1)
                logger.info(f"Applied OneHotEncoder to {col}, created {len(self.dummy_columns[col])} new columns")

            elif self.method == 'label':
                # Apply label encoding
                encoder = self.encoders[col]
                X_transformed[col] = encoder.transform(X_transformed[col])
                logger.info(f"Applied LabelEncoder to {col}")

            elif self.method == 'dummies':
                # Use pandas.get_dummies
                dummies = pd.get_dummies(X_transformed[col], prefix=col, drop_first=self.drop_first, dtype=int)

                # Drop original column and join dummy columns
                X_transformed = X_transformed.drop(columns=[col])
                X_transformed = pd.concat([X_transformed, dummies], axis=1)

                # Check for any missing dummy columns that were in the training data
                expected_dummy_cols = set(self.dummy_columns[col])
                actual_dummy_cols = set([col for col in dummies.columns])

                # Add missing dummy columns (with all zeros)
                for missing_col in expected_dummy_cols - actual_dummy_cols:
                    X_transformed[missing_col] = 0

                logger.info(f"Applied pd.get_dummies to {col}, created {dummies.shape[1]} new columns")

        return X_transformed


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """A transformer that returns the data unchanged."""

    def __init__(self):
        self.logger = logging.getLogger("Feature Engineering")
        self.logger.info("Initializing IdentityTransformer")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class FeatureToolsTransformer(BaseEstimator, TransformerMixin):
    """Enhanced transformer that uses featuretools aggressively for numeric data."""

    def __init__(self, target_col: str, max_features: int = 50000, chunk_size: int = 5000):
        self.logger = logging.getLogger("Feature Engineering")
        self.logger.info(f"Initializing Enhanced FeatureToolsTransformer with max_features={max_features}")
        self.target_col = target_col
        self.max_features = max_features
        self.chunk_size = chunk_size
        self.feature_defs = None
        self.feature_names = None
        self.selected_feature_defs = None

        try:
            import featuretools as ft
            self.ft = ft
            # List available primitives
            self.logger.info("Loading all available featuretools primitives...")
            self.available_primitives = self._get_all_primitives()
        except ImportError:
            self.logger.error("Featuretools package not found. Please install with: pip install featuretools")
            raise

    def _get_all_primitives(self):
        """Get comprehensive list of all available primitives for numeric data."""
        try:
            # Get all available primitives
            all_primitives_df = self.ft.primitives.list_primitives()

            # Get transform primitives (these work on individual rows/entities)
            transform_primitives = []

            # Extract primitive names from the dataframe
            for idx, row in all_primitives_df.iterrows():
                if row['type'] == 'transform':
                    prim_name = row['name']
                    try:
                        # Try to get the actual primitive class
                        prim_class = getattr(self.ft.primitives, prim_name, None)
                        if prim_class is not None:
                            transform_primitives.append(prim_name)
                    except:
                        # If we can't get the class, still add the name
                        transform_primitives.append(prim_name)

            # Also include some known good primitives manually
            known_good_primitives = [
                # Basic arithmetic
                'add_numeric', 'subtract_numeric', 'multiply_numeric', 'divide_numeric',
                'modulo_numeric', 'absolute', 'negate',

                # Mathematical functions
                'square_root', 'natural_logarithm', 'exponential', 'square',
                'cube_root', 'sine', 'cosine', 'tangent',

                # Statistical transforms
                'diff', 'rolling_mean', 'rolling_max', 'rolling_min', 'rolling_std',
                'cum_sum', 'cum_mean', 'cum_max', 'cum_min', 'cum_count',

                # Percentile and ranking
                'percentile', 'rank',

                # Time-based (if applicable)
                'lag', 'lead',

                # Comparison
                'greater_than', 'less_than', 'equal',

                # Binning/discretization
                'binary_threshold', 'percentile_bucket',

                # Advanced mathematical
                'log_1p', 'sigmoid', 'tanh'
            ]

            # Combine and deduplicate
            all_transform_primitives = list(set(transform_primitives + known_good_primitives))

            self.logger.info(f"Found {len(all_transform_primitives)} transform primitives")
            self.logger.info(f"Sample primitives: {all_transform_primitives[:20]}")

            return all_transform_primitives

        except Exception as e:
            self.logger.error(f"Error getting primitives: {e}")
            # Fallback to basic set
            return ['add_numeric', 'subtract_numeric', 'multiply_numeric', 'divide_numeric',
                    'absolute', 'square_root', 'natural_logarithm', 'exponential']

    def _filter_working_primitives(self, primitives, sample_data):
        """Test primitives on sample data to see which ones actually work."""
        working_primitives = []

        self.logger.info(f"Testing {len(primitives)} primitives on sample data...")

        for prim_name in primitives:
            try:
                # Create a small entityset for testing
                es_test = self.ft.EntitySet(id="test")
                sample_df = sample_data.head(100).copy()

                if sample_df.index.name is None:
                    sample_df = sample_df.reset_index(drop=True)
                    sample_df.index.name = "index"

                es_test = es_test.add_dataframe(
                    dataframe_name="test_data",
                    dataframe=sample_df,
                    index="index"
                )

                # Try to generate features with this primitive
                _, _ = self.ft.dfs(
                    entityset=es_test,
                    target_dataframe_name="test_data",
                    trans_primitives=[prim_name],
                    max_depth=1,
                    verbose=False,
                    max_features=10  # Just test with a few features
                )

                working_primitives.append(prim_name)

            except Exception as e:
                # Primitive doesn't work, skip it
                continue

        self.logger.info(f"Found {len(working_primitives)} working primitives: {working_primitives}")
        return working_primitives

    def _select_best_features(self, feature_matrix: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Select the best features using multiple criteria."""
        self.logger.info(
            f"Selecting best {self.max_features} features from {feature_matrix.shape[1]} generated features")

        # Remove features with zero or very low variance
        numeric_features = feature_matrix.select_dtypes(include=[np.number])

        # Calculate variance and remove constant features
        variances = numeric_features.var()
        variance_threshold = variances.quantile(0.1)  # Keep top 90% by variance
        high_variance_mask = variances > variance_threshold
        high_variance_features = numeric_features.loc[:, high_variance_mask]

        self.logger.info(f"Retained {high_variance_features.shape[1]} features after variance filtering")

        if high_variance_features.shape[1] <= self.max_features:
            return high_variance_features

        # Use multiple selection criteria
        feature_scores = {}

        # 1. Variance score (normalized)
        variances_norm = (variances - variances.min()) / (variances.max() - variances.min())

        # 2. Correlation with target (if available)
        if y is not None:
            try:
                correlations = high_variance_features.corrwith(y).abs()
                correlations = correlations.fillna(0)
                correlations_norm = correlations / correlations.max() if correlations.max() > 0 else correlations
            except Exception as e:
                self.logger.warning(f"Could not compute correlations with target: {str(e)}")
                correlations_norm = pd.Series(0, index=high_variance_features.columns)
        else:
            correlations_norm = pd.Series(0, index=high_variance_features.columns)

        # 3. Feature uniqueness (prefer features with more unique values)
        uniqueness_scores = high_variance_features.nunique() / len(high_variance_features)
        uniqueness_norm = uniqueness_scores / uniqueness_scores.max() if uniqueness_scores.max() > 0 else uniqueness_scores

        # Combine scores with weights
        for col in high_variance_features.columns:
            variance_score = variances_norm.get(col, 0)
            correlation_score = correlations_norm.get(col, 0)
            uniqueness_score = uniqueness_norm.get(col, 0)

            # Weighted combination - emphasize correlation with target
            combined_score = (0.2 * variance_score +
                              0.6 * correlation_score +
                              0.2 * uniqueness_score)
            feature_scores[col] = combined_score

        # Select top features
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [col for col, score in sorted_features[:self.max_features]]

        self.logger.info(f"Selected top {len(selected_features)} features based on combined scoring")

        return high_variance_features[selected_features]

    def fit(self, X, y=None):
        try:
            self.logger.info("Starting aggressive feature generation for numeric data...")

            # Ensure X is a DataFrame
            if isinstance(X, pd.DataFrame):
                X_copy = X.copy()
            else:
                X_array = np.asarray(X)
                if X_array.ndim > 2:
                    X_array = X_array.reshape(X_array.shape[0], -1)
                X_copy = pd.DataFrame(X_array)

            # Ensure y is properly formatted
            if y is not None:
                if hasattr(y, 'values'):
                    y = y.values
                y = np.asarray(y).flatten()

            # Since data is all numeric, we can be aggressive
            self.logger.info(f"Input data shape: {X_copy.shape}")
            self.logger.info(
                f"All columns are numeric: {X_copy.select_dtypes(include=[np.number]).shape[1] == X_copy.shape[1]}")

            # Test primitives on sample data to find working ones
            working_primitives = self._filter_working_primitives(self.available_primitives, X_copy)

            if not working_primitives:
                self.logger.warning("No working primitives found, using original features")
                self.feature_names = X_copy.columns.tolist()
                return self

            # Create entityset
            es = self.ft.EntitySet(id="aggressive_features")
            X_indexed = X_copy.copy()

            # Ensure we have a proper index
            if X_indexed.index.name is None:
                X_indexed = X_indexed.reset_index(drop=True)
                X_indexed.index.name = "index"

            # Add dataframe to entityset
            es = es.add_dataframe(
                dataframe_name="data",
                dataframe=X_indexed,
                index="index"
            )

            # Use aggressive parameters for maximum feature generation
            max_depth = 2  # Increase depth for more complex features

            self.logger.info(f"Generating features with max_depth={max_depth}")
            self.logger.info(f"Using {len(working_primitives)} primitives: {working_primitives[:10]}...")

            try:
                # Generate features aggressively
                feature_matrix, feature_defs = self.ft.dfs(
                    entityset=es,
                    target_dataframe_name="data",
                    trans_primitives=working_primitives,
                    max_depth=max_depth,
                    verbose=True,
                    max_features=None,  # No initial limit - generate as many as possible
                    n_jobs=1  # Single job to avoid memory issues
                )

                self.logger.info(f"Successfully generated {feature_matrix.shape[1]} raw features!")

                # Clean the feature matrix
                feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)

                # Remove columns with all NaN values
                feature_matrix = feature_matrix.dropna(axis=1, how='all')

                # Remove target column if present
                feature_columns = [col for col in feature_matrix.columns
                                   if self.target_col not in str(col)]

                if feature_columns:
                    feature_matrix_clean = feature_matrix[feature_columns]
                    self.logger.info(f"After cleaning: {feature_matrix_clean.shape[1]} features")
                else:
                    self.logger.warning("No valid features after cleaning, using original features")
                    self.feature_names = X_copy.columns.tolist()
                    return self

                # Select best features using comprehensive scoring
                selected_features = self._select_best_features(feature_matrix_clean,
                                                               pd.Series(y) if y is not None else None)
                self.feature_names = selected_features.columns.tolist()

                # Store feature definitions for selected features
                if feature_defs:
                    selected_feature_names = set(self.feature_names)
                    self.selected_feature_defs = []

                    for fd in feature_defs:
                        feature_name = fd.get_name() if hasattr(fd, 'get_name') else str(fd)
                        if feature_name in selected_feature_names:
                            self.selected_feature_defs.append(fd)

                    self.logger.info(f"Stored {len(self.selected_feature_defs)} feature definitions")
                else:
                    self.selected_feature_defs = []

                self.logger.info(f"Final feature count: {len(self.feature_names)}")
                self.logger.info(f"Sample generated features: {self.feature_names[:10]}")

            except Exception as dfs_error:
                self.logger.error(f"Aggressive DFS failed: {str(dfs_error)}")

                # Try with reduced primitives
                self.logger.info("Retrying with reduced primitive set...")

                # Use most reliable primitives
                safe_primitives = [p for p in ['add_numeric', 'subtract_numeric', 'multiply_numeric',
                                               'divide_numeric', 'absolute', 'square_root']
                                   if p in working_primitives]

                if safe_primitives:
                    try:
                        feature_matrix, feature_defs = self.ft.dfs(
                            entityset=es,
                            target_dataframe_name="data",
                            trans_primitives=safe_primitives,
                            max_depth=2,  # Still use depth 2
                            verbose=False,
                            max_features=self.max_features * 2  # Generate 2x target for selection
                        )

                        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
                        feature_matrix = feature_matrix.dropna(axis=1, how='all')

                        feature_columns = [col for col in feature_matrix.columns
                                           if self.target_col not in str(col)]

                        if feature_columns:
                            feature_matrix_clean = feature_matrix[feature_columns]
                            selected_features = self._select_best_features(feature_matrix_clean,
                                                                           pd.Series(y) if y is not None else None)
                            self.feature_names = selected_features.columns.tolist()
                            self.selected_feature_defs = feature_defs[:len(self.feature_names)]
                        else:
                            self.feature_names = X_copy.columns.tolist()
                            self.selected_feature_defs = []

                    except Exception as safe_error:
                        self.logger.error(f"Even safe primitive generation failed: {str(safe_error)}")
                        self.feature_names = X_copy.columns.tolist()
                        self.selected_feature_defs = []
                else:
                    self.feature_names = X_copy.columns.tolist()
                    self.selected_feature_defs = []

            return self

        except Exception as e:
            self.logger.error(f"Complete failure in FeatureToolsTransformer fit: {str(e)}")
            # Final fallback to original features
            self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"feature_{i}" for i in
                                                                                   range(X.shape[1])]
            self.selected_feature_defs = []
            return self

    def transform(self, X):
        """Transform data using stored feature definitions."""
        try:
            if not self.feature_names:
                self.logger.warning("No feature names available, returning original data")
                return X

            # If no valid feature definitions, return original features
            if not self.selected_feature_defs:
                self.logger.info("No feature definitions available, returning original features")
                if hasattr(X, 'columns'):
                    available_features = [col for col in self.feature_names if col in X.columns]
                    if available_features:
                        return X[available_features]
                return X

            # Regenerate features using stored definitions
            X_copy = X.copy()

            if X_copy.index.name is None:
                X_copy = X_copy.reset_index(drop=True)
                X_copy.index.name = "index"

            es = self.ft.EntitySet(id="features_transform")
            es = es.add_dataframe(
                dataframe_name="data",
                dataframe=X_copy,
                index="index"
            )

            try:
                # Calculate features using stored definitions
                feature_matrix = self.ft.calculate_feature_matrix(
                    features=self.selected_feature_defs,
                    entityset=es,
                    verbose=False
                )

                # Clean the results
                feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)

                # Handle NaN values - fill with 0 for now
                feature_matrix = feature_matrix.fillna(0)

                # Ensure we have numeric data
                feature_matrix = feature_matrix.select_dtypes(include=[np.number])

                # Filter to selected features
                available_features = [col for col in feature_matrix.columns if col in self.feature_names]

                if available_features:
                    result = feature_matrix[available_features]
                    self.logger.info(f"Generated {result.shape[1]} features for transform")
                    return result
                else:
                    self.logger.warning("No matching features found in generated matrix")
                    return X

            except Exception as calc_error:
                self.logger.error(f"Feature calculation failed during transform: {str(calc_error)}")
                # Return original features
                if hasattr(X, 'columns'):
                    available_features = [col for col in self.feature_names if col in X.columns]
                    if available_features:
                        return X[available_features]
                return X

        except Exception as e:
            self.logger.error(f"Error in FeatureToolsTransformer transform: {str(e)}")
            return X


class SHAPFeatureSelector(BaseEstimator, TransformerMixin):
    """A memory-efficient transformer that uses SHAP values to select important features."""

    def __init__(self, n_features: int = 20, model=None, max_samples_for_shap: int = 1000):
        self.logger = logging.getLogger("Feature Engineering")
        self.logger.info(f"Initializing SHAPFeatureSelector with n_features={n_features}")
        self.n_features = n_features
        self.max_samples_for_shap = max_samples_for_shap
        self.selected_features = None
        self.importance_df = None
        self.model = model or RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_scores = {}

        try:
            import shap
            self.shap = shap
        except ImportError:
            self.logger.error("SHAP package not found. Please install with: pip install shap")
            raise

    def _prefilter_features(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Pre-filter features using statistical methods to reduce dimensionality before SHAP."""
        self.logger.info(f"Pre-filtering features from {X.shape[1]} to manageable number")

        # Only work with numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        if X_numeric.shape[1] == 0:
            self.logger.warning("No numeric columns found for pre-filtering")
            return X

        # Clean the data
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
        X_numeric = X_numeric.fillna(X_numeric.median())

        feature_scores = {}

        # 1. Variance filtering
        variances = X_numeric.var()
        variance_scores = (variances - variances.min()) / (variances.max() - variances.min())

        # 2. Correlation with target (if available)
        if y is not None:
            try:
                correlations = X_numeric.corrwith(y).abs()
                correlation_scores = correlations.fillna(0)
            except:
                correlation_scores = pd.Series(0, index=X_numeric.columns)
        else:
            correlation_scores = pd.Series(0, index=X_numeric.columns)

        # 3. Mutual information (simplified version using correlation)
        # For large datasets, we'll use correlation as a proxy

        # Combine scores
        for col in X_numeric.columns:
            variance_score = variance_scores.get(col, 0)
            correlation_score = correlation_scores.get(col, 0)

            # Weighted combination
            combined_score = 0.3 * variance_score + 0.7 * correlation_score
            feature_scores[col] = combined_score

        # Select top features for SHAP analysis
        # Use 5x the target number to give SHAP good options
        n_prefilter = min(self.n_features * 5, 1000, X_numeric.shape[1])

        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:n_prefilter]
        selected_cols = [col for col, score in top_features]

        self.logger.info(f"Pre-filtered to {len(selected_cols)} features for SHAP analysis")
        return X_numeric[selected_cols]

    def fit(self, X, y=None):
        try:
            # Ensure X is a DataFrame
            if isinstance(X, pd.DataFrame):
                X_copy = X.copy()
            else:
                X_array = np.asarray(X)
                if X_array.ndim > 2:
                    X_array = X_array.reshape(X_array.shape[0], -1)
                X_copy = pd.DataFrame(X_array)

            # Ensure y is properly formatted
            if y is not None:
                if hasattr(y, 'values'):
                    y = y.values
                y = np.asarray(y).flatten()

            # Adjust n_features to available features
            self.n_features = min(self.n_features, X_copy.shape[1])

            # Handle small datasets
            if X_copy.shape[0] < 10:
                self.logger.warning("Dataset too small for SHAP analysis, selecting all features")
                self.selected_features = X_copy.columns.tolist()
                return self

            # Pre-filter features if we have too many
            if X_copy.shape[1] > 1000:
                self.logger.info(f"Large number of features ({X_copy.shape[1]}), applying pre-filtering")
                X_filtered = self._prefilter_features(X_copy, pd.Series(y) if y is not None else None)
            else:
                # Only work with numeric columns
                numeric_cols = X_copy.select_dtypes(include=[np.number]).columns
                X_filtered = X_copy[numeric_cols]

            if X_filtered.shape[1] == 0:
                self.logger.warning("No numeric columns found, selecting all original features")
                self.selected_features = X_copy.columns.tolist()
                return self

            # Clean the data
            X_filtered = X_filtered.replace([np.inf, -np.inf], np.nan)
            X_filtered = X_filtered.fillna(X_filtered.median())

            # Sample data if too large for SHAP
            if X_filtered.shape[0] > self.max_samples_for_shap:
                self.logger.info(f"Sampling {self.max_samples_for_shap} rows for SHAP analysis")
                sample_idx = np.random.choice(X_filtered.shape[0], self.max_samples_for_shap, replace=False)
                X_shap = X_filtered.iloc[sample_idx]
                y_shap = y[sample_idx] if y is not None else None
            else:
                X_shap = X_filtered
                y_shap = y

            # Try to fit the model with error handling
            try:
                self.model.fit(X_shap, y_shap)
            except Exception as model_error:
                self.logger.warning(f"Primary model failed: {str(model_error)}, using LogisticRegression")
                try:
                    self.model = LogisticRegression(max_iter=1000, random_state=42)
                    self.model.fit(X_shap, y_shap)
                except Exception as lr_error:
                    self.logger.error(f"LogisticRegression also failed: {str(lr_error)}")
                    # Fallback: select features based on variance
                    feature_vars = X_filtered.var().sort_values(ascending=False)
                    self.selected_features = feature_vars.head(self.n_features).index.tolist()
                    return self

            # Compute SHAP values with proper error handling
            try:
                # Use smaller sample for SHAP computation if needed
                shap_sample_size = min(500, X_shap.shape[0])
                X_shap_sample = X_shap.sample(n=shap_sample_size, random_state=42) if X_shap.shape[
                                                                                          0] > shap_sample_size else X_shap

                # Use TreeExplainer for tree-based models
                if hasattr(self.model, 'estimators_') or hasattr(self.model, 'n_estimators'):
                    explainer = self.shap.TreeExplainer(self.model)
                    shap_values = explainer.shap_values(X_shap_sample)
                else:
                    # Use KernelExplainer for other models with smaller background
                    background_size = min(100, X_shap_sample.shape[0] // 2)
                    background = X_shap_sample.sample(n=background_size, random_state=42)
                    explainer = self.shap.KernelExplainer(self.model.predict_proba, background)

                    # Use even smaller sample for explanation
                    explain_size = min(200, X_shap_sample.shape[0])
                    X_explain = X_shap_sample.sample(n=explain_size, random_state=42)
                    shap_values = explainer.shap_values(X_explain)

                # Process SHAP values
                if isinstance(shap_values, list):
                    # Multi-class case: average across classes
                    shap_importances = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                else:
                    # Binary case
                    shap_importances = np.abs(shap_values).mean(axis=0)

                # Create importance DataFrame
                self.importance_df = pd.DataFrame({
                    'feature': X_shap_sample.columns,
                    'importance': shap_importances
                }).sort_values('importance', ascending=False)

                # Select top features
                self.selected_features = self.importance_df['feature'].head(self.n_features).tolist()

                self.logger.info(f"SHAP feature selection completed. Selected {len(self.selected_features)} features")

                # Log top features
                if len(self.selected_features) <= 20:
                    self.logger.info(f"Selected features: {self.selected_features}")
                else:
                    self.logger.info(f"Top 10 selected features: {self.selected_features[:10]}")

            except Exception as shap_error:
                self.logger.warning(f"SHAP computation failed: {str(shap_error)}, using variance-based selection")
                # Fallback to variance-based feature selection
                feature_vars = X_filtered.var().sort_values(ascending=False)
                self.selected_features = feature_vars.head(self.n_features).index.tolist()

            return self

        except Exception as e:
            self.logger.error(f"SHAP fit failed completely: {str(e)}")
            # Final fallback: use all available features or first n features
            if isinstance(X, pd.DataFrame):
                all_features = X.columns.tolist()
            else:
                all_features = [f"feature_{i}" for i in range(X.shape[1])]

            self.selected_features = all_features[:min(len(all_features), self.n_features)]
            return self

    def transform(self, X):
        """Transform the data by selecting only the important features."""
        try:
            if self.selected_features is None:
                self.logger.warning("No features selected, returning original data")
                return X

            if isinstance(X, pd.DataFrame):
                # For DataFrame input, select available features
                available_features = [f for f in self.selected_features if f in X.columns]
                if not available_features:
                    self.logger.warning("No selected features found in input data, returning first n columns")
                    return X.iloc[:, :min(len(self.selected_features), X.shape[1])]

                result = X[available_features]
                self.logger.info(f"Selected {len(available_features)} features from {X.shape[1]} available features")
                return result
            else:
                # For numpy array input, assume feature order is preserved
                X_array = np.asarray(X)
                if X_array.ndim > 2:
                    X_array = X_array.reshape(X_array.shape[0], -1)

                # Select features by index (assuming feature names are like "feature_0", "feature_1", etc.)
                if all(f.startswith('feature_') for f in self.selected_features):
                    indices = [int(f.split('_')[1]) for f in self.selected_features if f.split('_')[1].isdigit()]
                    indices = [i for i in indices if i < X_array.shape[1]]
                    if indices:
                        return pd.DataFrame(X_array[:, indices])

                # Fallback: return first n_features columns
                n_cols = min(len(self.selected_features), X_array.shape[1])
                return pd.DataFrame(X_array[:, :n_cols])

        except Exception as e:
            self.logger.error(f"Error in SHAPFeatureSelector transform: {str(e)}")
            # Return original data as fallback
            return X