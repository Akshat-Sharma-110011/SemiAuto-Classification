"""
Data Preprocessing Module for SemiAuto-Regression

This module handles the preprocessing of data for the regression model, including:
- Handling missing values
- Handling duplicate values
- Handling outliers
- Handling skewed data
- Scaling numerical features
- Encoding categorical features

The preprocessing steps are configured based on information in the feature_store.yaml file,
and the preprocessing pipeline is saved for later use.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import cloudpickle
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder
)
import scipy.stats as stats
from pydantic import BaseModel

# Set up the logger
from semiauto_classification.logger import section, configure_logger
from semiauto_classification.custom_transformers import (IDColumnDropper, MissingValueHandler,
    CategoricalEncoder, NumericalScaler, SkewedDataHandler, OutlierHandler)
# Configure logger
configure_logger()
logger = logging.getLogger("Data Preprocessing")

def get_dataset_name():
    """Lazily load dataset name when needed"""
    try:
        with open('intel.yaml', 'r') as f:
            config = yaml.safe_load(f)
            return config['dataset_name']
    except FileNotFoundError:
        return "default_dataset"  # Temporary placeholder

dataset_name = get_dataset_name()



class PreprocessingParameters(BaseModel):
    """Pydantic model for preprocessing parameters"""
    # Missing Values Handling
    missing_values_method: str = 'mean'
    missing_values_columns: List[str] = []

    # Duplicates Handling
    handle_duplicates: bool = True

    # Outlier Handling
    outliers_method: Optional[str] = None
    outliers_columns: List[str] = []

    # Skewness Handling
    skewness_method: Optional[str] = None
    skewness_columns: List[str] = []

    # Numerical Scaling
    scaling_method: Optional[str] = None
    scaling_columns: List[str] = []

    # Categorical Encoding
    categorical_encoding_method: Optional[str] = None
    categorical_columns: List[str] = []
    drop_first: bool = True


class PreprocessingPipeline:
    """API-friendly preprocessing pipeline with configuration based on parameters"""

    def __init__(self, config: Dict[str, Any], params: PreprocessingParameters):
        """
        Initialize with both original config and API parameters

        Args:
            config: Original configuration from intel.yaml
            params: Preprocessing parameters from API request
        """
        self.config = get_intel_config()
        self.params = params
        self.dataset_name = config.get('dataset_name')
        self.target_column = config.get('target_col')
        self.feature_store = config.get('feature_store', {})

        # Initialize handlers
        self.missing_handler = None
        self.outlier_handler = None
        self.skewed_handler = None
        self.numerical_scaler = None
        self.categorical_encoder = None
        self.id_dropper = IDColumnDropper(
            id_cols=self.feature_store.get('id_cols', [])
        )
        self.target_encoder = None

        logger.info(f"Initialized API PreprocessingPipeline for dataset: {self.dataset_name}")

    def configure_pipeline(self):
        """Configure pipeline based on received parameters"""
        # Filter out target column from all preprocessing columns
        target_col = self.target_column

        # Missing values handling
        missing_cols = [col for col in self.params.missing_values_columns if col != target_col]
        if missing_cols:
            self.missing_handler = MissingValueHandler(
                method=self.params.missing_values_method,
                columns=missing_cols
            )

        # Initialize ID dropper with config from feature store
        self.id_dropper = IDColumnDropper(
            id_cols=self.feature_store.get('id_cols', [])
        )

        # Outlier handling
        outlier_cols = [col for col in self.params.outliers_columns if col != target_col]
        if self.params.outliers_method and outlier_cols:
            self.outlier_handler = OutlierHandler(
                method=self.params.outliers_method,
                columns=outlier_cols
            )

        # Skewed data handling
        skewed_cols = [col for col in self.params.skewness_columns if col != target_col]
        if self.params.skewness_method and skewed_cols:
            self.skewed_handler = SkewedDataHandler(
                method=self.params.skewness_method,
                columns=skewed_cols
            )

        # Numerical scaling
        scaling_cols = [col for col in self.params.scaling_columns if col != target_col]
        if self.params.scaling_method and scaling_cols:
            self.numerical_scaler = NumericalScaler(
                method=self.params.scaling_method,
                columns=scaling_cols
            )

        # Categorical encoding
        categorical_cols = [col for col in self.params.categorical_columns if col != target_col]
        if self.params.categorical_encoding_method and categorical_cols:
            self.categorical_encoder = CategoricalEncoder(
                method=self.params.categorical_encoding_method,
                columns=categorical_cols,
                drop_first=self.params.drop_first
            )

    def handle_missing_values(self, method: str = 'mean', columns: List[str] = None):
        if columns:
            columns = [col for col in columns if col != self.target_column]
        self.missing_handler = MissingValueHandler(method=method, columns=columns)
        logger.info(f"Set up missing value handler with method: {method}")

    def handle_outliers(self, method: str = 'IQR', columns: List[str] = None):
        if columns:
            columns = [col for col in columns if col != self.target_column]
        self.outlier_handler = OutlierHandler(method=method, columns=columns)
        logger.info(f"Set up outlier handler with method: {method}")

    def handle_skewed_data(self, method: str = 'yeo-johnson', columns: List[str] = None):
        if columns:
            columns = [col for col in columns if col != self.target_column]
        self.skewed_handler = SkewedDataHandler(method=method, columns=columns)
        logger.info(f"Set up skewed data handler with method: {method}")

    def scale_numerical_features(self, method: str = 'standard', columns: List[str] = None):
        if columns:
            columns = [col for col in columns if col != self.target_column]
        self.numerical_scaler = NumericalScaler(method=method, columns=columns)
        logger.info(f"Set up numerical scaler with method: {method}")

    def encode_categorical_features(self, method: str = 'onehot', columns: List[str] = None, drop_first: bool = True):
        """
        Set up the categorical encoder.

        Args:
            method (str): Method to use for encoding categorical features
            columns (List[str]): List of columns to encode
            drop_first (bool): Whether to drop the first category in one-hot encoding
        """
        self.categorical_encoder = CategoricalEncoder(method=method, columns=columns, drop_first=drop_first)
        logger.info(f"Set up categorical encoder with method: {method}")

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the data.

        Args:
            df (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Data with duplicates removed
        """
        rows_before = len(df)
        df_no_duplicates = df.drop_duplicates()
        rows_after = len(df_no_duplicates)
        rows_dropped = rows_before - rows_after

        if rows_dropped > 0:
            logger.info(f"Removed {rows_dropped} duplicate rows")
        else:
            logger.info("No duplicate rows found")

        return df_no_duplicates

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the preprocessing pipeline on the training data.

        Args:
            X (pd.DataFrame): Training data
        """
        logger.info("Fitting preprocessing pipeline")

        if self.id_dropper:
            self.id_dropper.fit(X)

        # Fit the missing value handler if defined
        if self.missing_handler:
            self.missing_handler.fit(X)

        # Fit the outlier handler if defined
        if self.outlier_handler:
            self.outlier_handler.fit(X)

        # Fit the skewed data handler if defined
        if self.skewed_handler:
            self.skewed_handler.fit(X)

        # Fit the numerical scaler if defined
        if self.numerical_scaler:
            self.numerical_scaler.fit(X)

        # Fit the categorical encoder if defined
        if self.categorical_encoder:
            self.categorical_encoder.fit(X)

        if self.target_column and self.target_column in X.columns:
            target_data = X[self.target_column]
            if not pd.api.types.is_numeric_dtype(target_data):
                self.target_encoder = LabelEncoder()
                self.target_encoder.fit(target_data)
                logger.info(f"Fitted LabelEncoder on target column '{self.target_column}'")

    def transform(self, X: pd.DataFrame, handle_duplicates: bool = True) -> pd.DataFrame:
        logger.info("Transforming data with preprocessing pipeline")
        transformed_data = X.copy()

        # Apply ID column dropper first
        if self.id_dropper:
            transformed_data = self.id_dropper.transform(transformed_data)

        # Handle duplicates if requested
        if handle_duplicates:
            transformed_data = self.remove_duplicates(transformed_data)

        # Extract target data before any transformations
        if self.target_column and self.target_column in transformed_data.columns:
            target_data = transformed_data.pop(self.target_column)
        else:
            target_data = None

        # Transform with missing value handler if defined
        if self.missing_handler:
            transformed_data = self.missing_handler.transform(transformed_data)

        # Transform with outlier handler if defined
        if self.outlier_handler:
            transformed_data = self.outlier_handler.transform(transformed_data)

        # Transform with skewed data handler if defined
        if self.skewed_handler:
            transformed_data = self.skewed_handler.transform(transformed_data)

        # Transform with numerical scaler if defined
        if self.numerical_scaler:
            transformed_data = self.numerical_scaler.transform(transformed_data)

        # Transform with categorical encoder if defined
        if self.categorical_encoder:
            transformed_data = self.categorical_encoder.transform(transformed_data)

            # Encode target if encoder is available
        if target_data is not None:
            # Encode target if encoder is available
            if self.target_encoder is not None:
                # Check for missing values in target
                if target_data.isnull().any():
                    logger.error("Target column contains missing values after preprocessing")
                    raise ValueError("Target column contains missing values")
                # Transform target
                encoded_target = self.target_encoder.transform(target_data)
                transformed_data[self.target_column] = encoded_target
                logger.info(f"Encoded target column '{self.target_column}' using LabelEncoder (original name preserved)")
            else:
                # Convert to numeric, coercing errors to NaN
                transformed_data[self.target_column] = pd.to_numeric(target_data, errors='coerce')
                if transformed_data[self.target_column].isnull().any():
                    logger.error("Target contains non-numeric values that couldn't be coerced")
                    raise ValueError("Invalid values in target column")
                logger.info(f"Converted target column '{self.target_column}' to numeric (original name preserved)")

        # Ensure target column is the last column
        if self.target_column and self.target_column in transformed_data.columns:
            cols = [col for col in transformed_data.columns if col != self.target_column] + [self.target_column]
            transformed_data = transformed_data[cols]

        return transformed_data

    def save(self, path: str) -> None:
        """
        Save the preprocessing pipeline using cloudpickle.

        Args:
            path (str): Path to save the pipeline
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            cloudpickle.dump(self, f)

        logger.info(f"Saved preprocessing pipeline to {path}")

    @classmethod
    def load(cls, path: str):
        """
        Load a preprocessing pipeline from a file.

        Args:
            path (str): Path to the pipeline file

        Returns:
            PreprocessingPipeline: Loaded pipeline
        """
        with open(path, 'rb') as f:
            pipeline = cloudpickle.load(f)

        logger.info(f"Loaded preprocessing pipeline from {path}")
        return pipeline


def validate_and_sanitize_parameters(params: PreprocessingParameters, feature_store: Dict) -> PreprocessingParameters:
    """Validate and sanitize preprocessing parameters against feature store"""
    validated = params.dict()

    # Validate missing values columns
    valid_missing = [col for col in validated['missing_values_columns']
                     if col in feature_store.get('contains_null', [])]
    validated['missing_values_columns'] = valid_missing

    # Validate outlier columns
    valid_outliers = [col for col in validated['outliers_columns']
                      if col in feature_store.get('contains_outliers', [])]
    validated['outliers_columns'] = valid_outliers

    # Validate skewness columns
    valid_skewness = [col for col in validated['skewness_columns']
                      if col in feature_store.get('skewed_cols', [])]
    validated['skewness_columns'] = valid_skewness

    return PreprocessingParameters(**validated)


async def api_preprocessing_workflow(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        params: PreprocessingParameters,
        config: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    API-friendly preprocessing workflow

    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        params: Preprocessing parameters
        config: Application config from intel.yaml

    Returns:
        Tuple of (train_preprocessed, test_preprocessed, preprocessing_config)
    """
    try:
        section("API PREPROCESSING WORKFLOW", logger)

        # Load feature store
        feature_store_path = config.get('feature_store_path')
        feature_store = load_yaml(feature_store_path)

        # Validate parameters against feature store
        validated_params = validate_and_sanitize_parameters(params, feature_store)

        # Initialize pipeline
        pipeline = PreprocessingPipeline(config, validated_params)
        pipeline.configure_pipeline()

        # Fit and transform
        section("FITTING PIPELINE", logger)
        pipeline.fit(train_df)

        section("TRANSFORMING DATA", logger)
        train_preprocessed = pipeline.transform(
            train_df,
            handle_duplicates=validated_params.handle_duplicates
        )
        test_preprocessed = pipeline.transform(
            test_df,
            handle_duplicates=False  # Never drop duplicates from test data
        )

        # Collect preprocessing config for response
        preprocessing_config = {
            'missing_values': {
                'method': validated_params.missing_values_method,
                'columns': validated_params.missing_values_columns
            },
            'outliers': {
                'method': validated_params.outliers_method,
                'columns': validated_params.outliers_columns
            },
            'skewness': {
                'method': validated_params.skewness_method,
                'columns': validated_params.skewness_columns
            },
            'scaling': {
                'method': validated_params.scaling_method,
                'columns': validated_params.scaling_columns
            },
            'categorical_encoding': {
                'method': validated_params.categorical_encoding_method,
                'columns': validated_params.categorical_columns,
                'drop_first': validated_params.drop_first
            }
        }

        # Move target column to last position if present
        target_column = config.get('target_col')
        if target_column in train_preprocessed.columns:
            cols = [col for col in train_preprocessed.columns if col != target_column] + [target_column]
            train_preprocessed = train_preprocessed[cols]
            test_preprocessed = test_preprocessed[cols]

        return train_preprocessed, test_preprocessed, preprocessing_config

    except Exception as e:
        logger.error(f"API Preprocessing failed: {str(e)}")
        raise


def save_preprocessing_artifacts(
        train_preprocessed: pd.DataFrame,
        test_preprocessed: pd.DataFrame,
        pipeline: PreprocessingPipeline,
        config: Dict
):
    """Save preprocessing results and pipeline"""
    try:
        # Create output directories
        interim_dir = Path(config.get('interim_dir', 'data/interim'))
        interim_dir.mkdir(parents=True, exist_ok=True)

        pipeline_dir = Path(config.get('pipeline_dir', 'model/pipelines'))
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        # Save preprocessed data
        train_preprocessed_path = interim_dir / 'train_preprocessed.csv'
        test_preprocessed_path = interim_dir / 'test_preprocessed.csv'
        train_preprocessed.to_csv(train_preprocessed_path, index=False)
        test_preprocessed.to_csv(test_preprocessed_path, index=False)

        # Save pipeline
        pipeline_path = pipeline_dir / 'preprocessing.pkl'
        pipeline.save(pipeline_path)

        # Update config
        config.update({
            'train_preprocessed_path': str(train_preprocessed_path),
            'test_preprocessed_path': str(test_preprocessed_path),
            'preprocessing_pipeline_path': str(pipeline_path),
            'preprocessed_timestamp': datetime.now().isoformat()
        })

        return config

    except Exception as e:
        logger.error(f"Failed to save preprocessing artifacts: {str(e)}")
        raise


def load_yaml(file_path: str) -> Dict:
    """
    Load YAML file into a dictionary.

    Args:
        file_path (str): Path to YAML file

    Returns:
        Dict: Loaded YAML content
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading YAML file {file_path}: {str(e)}")
        raise

def get_intel_config():
    """Safely load intel.yaml with validation"""
    try:
        with open('intel.yaml', 'r') as f:
            config = yaml.safe_load(f)
            # Validate critical keys
            if not config or 'dataset_name' not in config:
                raise ValueError("intel.yaml is missing required keys")
            return config
    except FileNotFoundError:
        raise RuntimeError("intel.yaml not found. Complete data ingestion first!")
    except Exception as e:
        raise RuntimeError(f"Invalid intel.yaml: {str(e)}")


def update_intel_yaml(intel_path: str, updates: Dict) -> None:
    """
    Update the intel.yaml file with new information.

    Args:
        intel_path (str): Path to intel.yaml file
        updates (Dict): Dictionary of updates to apply
    """
    try:
        # Load existing intel
        intel = load_yaml(intel_path)

        # Update with new information
        intel.update(updates)

        # Add processed timestamp
        intel['processed_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Write back to file
        with open(intel_path, 'w') as file:
            yaml.dump(intel, file, default_flow_style=False)

        logger.info(f"Updated intel.yaml at {intel_path}")
    except Exception as e:
        logger.error(f"Error updating intel.yaml: {str(e)}")
        raise


def check_for_duplicates(df: pd.DataFrame) -> bool:
    """
    Check if dataframe contains duplicate rows.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        bool: True if duplicates exist, False otherwise
    """
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.info(f"Found {duplicates} duplicate rows")
        return True
    else:
        logger.info("No duplicate rows found")
        return False


def check_for_skewness(df: pd.DataFrame, columns: List[str], threshold: float = 0.5) -> Dict[str, float]:
    """
    Check for skewness in the specified columns.

    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): Columns to check for skewness (should already exclude target)
        threshold (float): Skewness threshold (abs value) to consider a column skewed

    Returns:
        Dict[str, float]: Dictionary with column names as keys and skewness values as values
    """
    skewed_columns = {}

    for col in columns:
        if col not in df.columns:
            continue

        # Only process numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        skewness = df[col].skew()
        if abs(skewness) > threshold:
            skewed_columns[col] = skewness
            logger.info(f"Column {col} is skewed with skewness value: {skewness:.4f}")

    return skewed_columns


def get_numerical_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Get list of numerical columns in the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe
        exclude (List[str]): Columns to exclude

    Returns:
        List[str]: List of numerical columns
    """
    if exclude is None:
        exclude = []

    # Get columns with numeric dtype
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Exclude specified columns
    numeric_cols = [col for col in numeric_cols if col not in exclude]

    return numeric_cols


def get_categorical_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Get list of categorical columns in the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe
        exclude (List[str]): Columns to exclude

    Returns:
        List[str]: List of categorical columns
    """
    if exclude is None:
        exclude = []

    # Get columns with object or category dtype
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Exclude specified columns
    cat_cols = [col for col in cat_cols if col not in exclude]

    return cat_cols


def recommend_skewness_transformer(df: pd.DataFrame, column: str) -> str:
    """
    Recommend the best transformer for a skewed column.

    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to check

    Returns:
        str: Recommended transformer ('yeo-johnson' or 'box-cox')
    """
    # Safety check - don't process target column
    if not pd.api.types.is_numeric_dtype(df[column]):
        logger.warning(f"Column {column} is not numeric, defaulting to yeo-johnson")
        return 'yeo-johnson'

    # Check if column contains negative or zero values
    if df[column].min() <= 0:
        logger.info(f"Column {column} contains negative or zero values, recommending Yeo-Johnson transformation")
        return 'yeo-johnson'

    # Check the skewness after both transformations
    # Create a sample to test transformations (for speed)
    sample = df[column].sample(min(1000, len(df))).copy()

    # Test Yeo-Johnson
    try:
        yj_transformer = PowerTransformer(method='yeo-johnson')
        yj_transformed = yj_transformer.fit_transform(sample.values.reshape(-1, 1)).flatten()
        yj_skewness = stats.skew(yj_transformed)
    except Exception as e:
        logger.warning(f"Error testing Yeo-Johnson transformation: {str(e)}")
        yj_skewness = float('inf')

    # Test Box-Cox
    try:
        bc_transformer = PowerTransformer(method='box-cox')
        bc_transformed = bc_transformer.fit_transform(sample.values.reshape(-1, 1)).flatten()
        bc_skewness = stats.skew(bc_transformed)
    except Exception as e:
        logger.warning(f"Error testing Box-Cox transformation: {str(e)}")
        bc_skewness = float('inf')

    # Compare and recommend
    if abs(bc_skewness) <= abs(yj_skewness):
        logger.info(
            f"Box-Cox transformation recommended for {column} (skewness: {bc_skewness:.4f} vs {yj_skewness:.4f})")
        return 'box-cox'
    else:
        logger.info(f"Yeo-Johnson transformation recommended for {column} (skewness: {yj_skewness:.4f} vs {bc_skewness:.4f})")
        return 'yeo-johnson'


def main():
    """
    Main function to run the data preprocessing pipeline.
    """
    try:
        section("DATA PREPROCESSING", logger)
        logger.info("Starting data preprocessing")

        # Load intel.yaml
        intel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'intel.yaml')
        intel = load_yaml(intel_path)
        logger.info(f"Loaded intel from {intel_path}")

        # Extract information from intel.yaml
        dataset_name = intel.get('dataset_name')
        feature_store_path = intel.get('feature_store_path')
        train_path = intel.get('cleaned_train_path')
        test_path = intel.get('cleaned_test_path')
        target_column = intel.get('target_col')

        # Load feature store
        feature_store = load_yaml(feature_store_path)
        logger.info(f"Loaded feature store from {feature_store_path}")

        # Check for special columns in feature store and exclude target column
        null_columns = [col for col in feature_store.get('contains_null', []) if col != target_column]
        outlier_columns = [col for col in feature_store.get('contains_outliers', []) if col != target_column]
        skewed_columns = [col for col in feature_store.get('skewed_cols', []) if col != target_column]
        categorical_columns = [col for col in feature_store.get('categorical_cols', []) if col != target_column]

        # Load training and test data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info(f"Loaded training data: {train_df.shape} and test data: {test_df.shape}")

        # Initialize preprocessing pipeline
        pipeline_config = {
            'dataset_name': dataset_name,
            'target_col': target_column,
            'feature_store': feature_store
        }
        pipeline = PreprocessingPipeline(pipeline_config)

        # Check for duplicates in training data
        has_duplicates = check_for_duplicates(train_df)

        # If categorical columns not specified in feature store, detect them automatically
        if not categorical_columns:
            categorical_columns = get_categorical_columns(train_df, exclude=[target_column])
            logger.info(f"Auto-detected categorical columns: {categorical_columns}")

        # If skewed columns not specified in feature store, detect them automatically
        if not skewed_columns:
            numerical_columns = get_numerical_columns(train_df, exclude=[target_column])
            skewness_dict = check_for_skewness(train_df, numerical_columns)
            skewed_columns = list(skewness_dict.keys())
            logger.info(f"Auto-detected skewed columns: {skewed_columns}")

        # Set up paths for output files
        interim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'interim',
                                   f'data_{dataset_name}')
        os.makedirs(interim_dir, exist_ok=True)

        train_preprocessed_path = os.path.join(interim_dir, 'train_preprocessed.csv')
        test_preprocessed_path = os.path.join(interim_dir, 'test_preprocessed.csv')

        pipeline_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'pipelines',
                                    f'preprocessing_{dataset_name}')
        os.makedirs(pipeline_dir, exist_ok=True)

        pipeline_path = os.path.join(pipeline_dir, 'preprocessing.pkl')

        # Interactive preprocessing
        handle_duplicates = True  # Default is to handle duplicates

        # Handle missing values if any
        if null_columns:
            logger.info(f"Found columns with null values: {null_columns}")
            print(f"Found columns with null values: {null_columns}")
            print("How would you like to handle missing values?")
            print("1. Use mean (for numerical columns)")
            print("2. Use median (for numerical columns)")
            print("3. Use mode (most frequent value)")
            print("4. Drop rows with missing values")

            choice = input("Enter your choice (1-4): ")

            if choice == '1':
                pipeline.handle_missing_values(method='mean', columns=null_columns)
            elif choice == '2':
                pipeline.handle_missing_values(method='median', columns=null_columns)
            elif choice == '3':
                pipeline.handle_missing_values(method='mode', columns=null_columns)
            elif choice == '4':
                pipeline.handle_missing_values(method='drop', columns=null_columns)
            else:
                logger.warning("Invalid choice, using default (mean)")
                pipeline.handle_missing_values(method='mean', columns=null_columns)

        # Handle duplicates if any
        if has_duplicates:
            print("Found duplicate rows in the training data.")
            print("How would you like to handle duplicates?")
            print("1. Drop duplicates")
            print("2. Keep duplicates")

            choice = input("Enter your choice (1-2): ")

            if choice == '1':
                handle_duplicates = True
            elif choice == '2':
                handle_duplicates = False
            else:
                logger.warning("Invalid choice, using default (drop duplicates)")
                handle_duplicates = True

        # Handle outliers if any
        if outlier_columns:
            logger.info(f"Found columns with outliers: {outlier_columns}")
            print(f"Found columns with outliers: {outlier_columns}")
            print("How would you like to handle outliers?")
            print("1. Use IQR method (cap at Q1 - 1.5 * IQR and Q3 + 1.5 * IQR)")
            print("2. Use Z-Score method (cap at mean ± 3 standard deviations)")

            choice = input("Enter your choice (1-2): ")

            if choice == '1':
                pipeline.handle_outliers(method='IQR', columns=outlier_columns)
            elif choice == '2':
                pipeline.handle_outliers(method='Z-Score', columns=outlier_columns)
            else:
                logger.warning("Invalid choice, using default (IQR)")
                pipeline.handle_outliers(method='IQR', columns=outlier_columns)

        # Handle skewed data if any - FIXED: Use filtered skewed_columns
        if skewed_columns:
            logger.info(f"Found columns with skewed distributions: {skewed_columns}")
            print(f"Found columns with skewed distributions: {skewed_columns}")

            # Show recommended transformer for each skewed column
            print("\nRecommended transformers for each skewed column:")
            recommended_transformers = {}
            for col in skewed_columns:
                recommended = recommend_skewness_transformer(train_df, col)
                recommended_transformers[col] = recommended
                print(f"  - {col}: {recommended}")

            print("\nHow would you like to handle skewed data?")
            print("1. Use Yeo-Johnson transformation (works with negative values)")
            print("2. Use Box-Cox transformation (requires positive values)")
            print("3. Use recommended transformer for each column")

            choice = input("Enter your choice (1-3): ")

            if choice == '1':
                pipeline.handle_skewed_data(method='yeo-johnson', columns=skewed_columns)
            elif choice == '2':
                pipeline.handle_skewed_data(method='box-cox', columns=skewed_columns)
            elif choice == '3':
                # We'll use the recommended transformer
                counts = {'yeo-johnson': 0, 'box-cox': 0}
                for col, transformer in recommended_transformers.items():
                    counts[transformer] += 1

                # Use the most recommended transformer
                if counts['box-cox'] > counts['yeo-johnson']:
                    pipeline.handle_skewed_data(method='box-cox', columns=skewed_columns)
                else:
                    pipeline.handle_skewed_data(method='yeo-johnson', columns=skewed_columns)
            else:
                logger.warning("Invalid choice, using default (Yeo-Johnson)")
                pipeline.handle_skewed_data(method='yeo-johnson', columns=skewed_columns)

        # Scale numerical features - FIXED: Exclude target column
        numerical_columns = get_numerical_columns(train_df, exclude=[target_column])
        if numerical_columns:
            logger.info(f"Found numerical columns: {numerical_columns}")
            print(f"\nFound numerical columns: {numerical_columns}")
            print("Would you like to scale these numerical features?")
            print("1. Yes, use StandardScaler (mean=0, std=1)")
            print("2. Yes, use RobustScaler (median=0, IQR=1, robust to outliers)")
            print("3. Yes, use MinMaxScaler (scale to range [0,1])")
            print("4. No, do not scale numerical features")

            choice = input("Enter your choice (1-4): ")

            if choice == '1':
                pipeline.scale_numerical_features(method='standard', columns=numerical_columns)
            elif choice == '2':
                pipeline.scale_numerical_features(method='robust', columns=numerical_columns)
            elif choice == '3':
                pipeline.scale_numerical_features(method='minmax', columns=numerical_columns)
            elif choice == '4':
                logger.info("Skipping numerical feature scaling")
            else:
                logger.warning("Invalid choice, skipping numerical feature scaling")

        # Encode categorical features - FIXED: Use filtered categorical_columns
        if categorical_columns:
            logger.info(f"Found categorical columns: {categorical_columns}")
            print(f"\nFound categorical columns: {categorical_columns}")
            print("How would you like to encode these categorical features?")
            print("1. Use OneHotEncoder (sklearn)")
            print("2. Use pd.get_dummies (pandas)")
            print("3. Use LabelEncoder (convert to integers)")

            choice = input("Enter your choice (1-3): ")

            if choice == '1':
                # Ask about dropping first category
                drop_first = input("Drop first category to avoid multicollinearity? (y/n): ").lower() == 'y'
                pipeline.encode_categorical_features(method='onehot', columns=categorical_columns, drop_first=drop_first)
            elif choice == '2':
                # Ask about dropping first category
                drop_first = input("Drop first category to avoid multicollinearity? (y/n): ").lower() == 'y'
                pipeline.encode_categorical_features(method='dummies', columns=categorical_columns, drop_first=drop_first)
            elif choice == '3':
                pipeline.encode_categorical_features(method='label', columns=categorical_columns)
            else:
                logger.warning("Invalid choice, using default (OneHotEncoder)")
                pipeline.encode_categorical_features(method='onehot', columns=categorical_columns)

        # Collect preprocessing configuration from the pipeline
        preprocessing_config = {}

        # Missing values handling
        if pipeline.missing_handler:
            preprocessing_config['missing_values'] = {
                'method': pipeline.missing_handler.method,
                'columns': pipeline.missing_handler.columns
            }

        # Outlier handling
        if pipeline.outlier_handler:
            preprocessing_config['outliers'] = {
                'method': pipeline.outlier_handler.method,
                'columns': pipeline.outlier_handler.columns
            }

        # Skewed data handling
        if pipeline.skewed_handler:
            preprocessing_config['skewed_data'] = {
                'method': pipeline.skewed_handler.method,
                'columns': pipeline.skewed_handler.columns
            }

        # Numerical scaling
        if pipeline.numerical_scaler:
            preprocessing_config['numerical_scaling'] = {
                'method': pipeline.numerical_scaler.method,
                'columns': pipeline.numerical_scaler.columns
            }

        # Categorical encoding
        if pipeline.categorical_encoder:
            preprocessing_config['categorical_encoding'] = {
                'method': pipeline.categorical_encoder.method,
                'columns': pipeline.categorical_encoder.columns,
                'drop_first': pipeline.categorical_encoder.drop_first
            }

        # Duplicates handling
        preprocessing_config['handle_duplicates'] = handle_duplicates

        section("FITTING PIPELINE", logger)
        # Fit the pipeline on training data
        pipeline.fit(train_df)

        section("TRANSFORMING DATA", logger)
        # Transform training data
        train_preprocessed = pipeline.transform(train_df, handle_duplicates=handle_duplicates)
        # Ensure target column is the last column
        if target_column in train_preprocessed.columns:
            cols = [col for col in train_preprocessed.columns if col != target_column] + [target_column]
            train_preprocessed = train_preprocessed[cols]
        logger.info(f"Transformed training data: {train_preprocessed.shape}")

        # Transform test data (without handling duplicates in test data)
        test_preprocessed = pipeline.transform(test_df, handle_duplicates=False)
        # Ensure target column is the last column
        if target_column in test_preprocessed.columns:
            cols = [col for col in test_preprocessed.columns if col != target_column] + [target_column]
            test_preprocessed = test_preprocessed[cols]
        logger.info(f"Transformed test data: {test_preprocessed.shape}")

        train_preprocessed.to_csv(train_preprocessed_path, index=False)
        test_preprocessed.to_csv(test_preprocessed_path, index=False)

        # Save the pipeline
        pipeline.save(pipeline_path)

        # Update intel.yaml with new paths and preprocessing configuration
        updates = {
            'train_preprocessed_path': train_preprocessed_path,
            'test_preprocessed_path': test_preprocessed_path,
            'preprocessing_pipeline_path': pipeline_path,
            'preprocessing_config': preprocessing_config
        }
        update_intel_yaml(intel_path, updates)

        section("PREPROCESSING COMPLETE", logger)
        logger.info("Data preprocessing completed successfully")

    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()