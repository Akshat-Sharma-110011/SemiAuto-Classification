import os
import sys
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional, Union, BinaryIO
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
import io
import tempfile
import shutil
import re

# Add the parent directory to the system path to import logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from logger import section, configure_logger
except ImportError:
    # Fallback logger functions if not available
    def section(text, logger=None, char="-", length=50):
        message = f"\n{char * length}\n{text}\n{char * length}"
        if logger:
            logger.info(message)
        else:
            print(message)


    def configure_logger():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )

# Constants
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')
FEATURE_STORE_DIR = os.path.join(PROJECT_DIR, 'references')
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports/figures')
CORRELATION_THRESHOLD = 0.65
SKEWNESS_THRESHOLD = 0.5
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTLIER_THRESHOLD = 1.5  # IQR multiplier for outlier detection
ID_COLUMN_THRESHOLD = 0.9  # Threshold for unique value ratio to identify ID columns

# Text detection constants
MIN_TEXT_LENGTH = 10  # Minimum average text length to consider as textual
TEXT_COLUMN_THRESHOLD = 0.7  # Threshold for text-like content ratio
MAX_UNIQUE_RATIO_FOR_TEXT = 0.8  # Maximum unique ratio for text columns (to exclude IDs)


class DataIngestion:
    """
    Class for handling data ingestion from uploaded files, performing initial analysis,
    and creating feature store metadata.

    Supports both tabular and textual data processing modes.
    """

    def __init__(self, project_dir=None):
        """
        Initialize the DataIngestion class

        Args:
            project_dir: Optional custom project directory path
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing DataIngestion component")

        # Use custom project dir if specified, otherwise use default
        self.project_dir = project_dir or PROJECT_DIR

        # Update paths based on project directory
        self.raw_data_dir = os.path.join(self.project_dir, 'data', 'raw')
        self.feature_store_dir = os.path.join(self.project_dir, 'references')
        self.reports_dir = os.path.join(self.project_dir, 'reports/figures')

        # Ensure directories exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.feature_store_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        self.df = None
        self.dataset_name = None
        self.processing_mode = "tabular"  # Default mode

        self.feature_store_data = {
            'original_cols': [],
            'numerical_cols': [],
            'categorical_cols': [],
            'textual_cols': [],  # New: textual columns
            'id_cols': [],
            'skewed_cols': [],
            'normal_cols': [],
            'contains_null': [],
            'contains_outliers': [],
            'correlated_cols': {},
            'target_col': None,
            'processing_mode': 'tabular',  # New: processing mode
            'timestamp': datetime.now().strftime(DATETIME_FORMAT),
            'train_size': 1 - TEST_SIZE,
            'test_size': TEST_SIZE
        }

    def detect_processing_mode(self) -> str:
        """
        Automatically detect the processing mode based on data characteristics

        Returns:
            str: 'tabular', 'textual', or 'mixed'
        """
        section("DETECTING PROCESSING MODE", self.logger)

        if self.df is None:
            self.logger.error("No data loaded. Please load data first.")
            return "tabular"

        # Get basic column analysis
        total_cols = len(self.df.columns)
        text_cols = self._identify_textual_columns()
        numerical_cols = self.df.select_dtypes(include=['number']).columns.tolist()

        text_ratio = len(text_cols) / total_cols if total_cols > 0 else 0
        numerical_ratio = len(numerical_cols) / total_cols if total_cols > 0 else 0

        self.logger.info(f"Text columns ratio: {text_ratio:.2f}")
        self.logger.info(f"Numerical columns ratio: {numerical_ratio:.2f}")

        # Decision logic
        if text_ratio >= 0.5:
            mode = "textual"
        elif text_ratio > 0.2 and numerical_ratio > 0.3:
            mode = "mixed"
        else:
            mode = "tabular"

        self.logger.info(f"Detected processing mode: {mode}")
        return mode

    def set_processing_mode(self, mode: str) -> str:
        """
        Set the processing mode manually

        Args:
            mode: 'tabular', 'textual', or 'mixed'

        Returns:
            str: The set processing mode
        """
        valid_modes = ['tabular', 'textual', 'mixed']
        if mode not in valid_modes:
            self.logger.warning(f"Invalid mode '{mode}'. Using 'tabular' as default.")
            mode = 'tabular'

        self.processing_mode = mode
        self.feature_store_data['processing_mode'] = mode
        self.logger.info(f"Processing mode set to: {mode}")
        return mode

    def _is_text_column(self, series: pd.Series) -> bool:
        """
        Determine if a pandas Series contains textual data

        Args:
            series: Pandas Series to analyze

        Returns:
            bool: True if column contains textual data
        """
        # Skip if mostly null
        if series.isnull().sum() / len(series) > 0.8:
            return False

        # Convert to string and drop nulls for analysis
        text_series = series.dropna().astype(str)

        if len(text_series) == 0:
            return False

        # Check average length
        avg_length = text_series.str.len().mean()
        if avg_length < MIN_TEXT_LENGTH:
            return False

        # Check for text-like patterns
        text_like_count = 0
        total_count = min(len(text_series), 1000)  # Sample for performance

        for text in text_series.head(total_count):
            # Check if contains multiple words
            word_count = len(str(text).split())
            # Check if contains letters
            has_letters = bool(re.search(r'[a-zA-Z]', str(text)))
            # Check if not purely numeric
            is_not_numeric = not str(text).replace('.', '').replace('-', '').isdigit()

            if word_count >= 2 and has_letters and is_not_numeric:
                text_like_count += 1

        text_ratio = text_like_count / total_count

        # Check uniqueness (to avoid ID columns)
        unique_ratio = series.nunique() / len(series)

        return (
                text_ratio >= TEXT_COLUMN_THRESHOLD and
                avg_length >= MIN_TEXT_LENGTH and
                (
                        unique_ratio <= MAX_UNIQUE_RATIO_FOR_TEXT  # normal text condition
                        or text_ratio > 0.9  # free-text override
                )
        )

    def _identify_textual_columns(self) -> List[str]:
        """
        Identify textual columns in the dataset

        Returns:
            List of column names that contain textual data
        """
        if self.df is None:
            return []

        textual_cols = []

        # Only check string/object columns
        object_cols = self.df.select_dtypes(include=['object', 'string']).columns

        for col in object_cols:
            if self._is_text_column(self.df[col]):
                textual_cols.append(col)
                self.logger.info(f"  - {col}: Identified as textual column")

        return textual_cols

    def analyze_text_characteristics(self) -> Dict[str, Dict]:
        """
        Analyze characteristics of textual columns

        Returns:
            Dictionary with text analysis results
        """
        section("ANALYZING TEXT CHARACTERISTICS", self.logger)

        text_analysis = {}
        textual_cols = self.feature_store_data.get('textual_cols', [])

        for col in textual_cols:
            try:
                # Convert to string and drop nulls
                text_series = self.df[col].dropna().astype(str)

                # Basic statistics
                analysis = {
                    'total_texts': len(text_series),
                    'unique_texts': text_series.nunique(),
                    'avg_length': text_series.str.len().mean(),
                    'max_length': text_series.str.len().max(),
                    'min_length': text_series.str.len().min(),
                    'avg_word_count': text_series.str.split().str.len().mean(),
                    'contains_punctuation': text_series.str.contains(r'[.!?]').mean(),
                    'avg_sentence_count': text_series.str.split(r'[.!?]').str.len().mean()
                }

                # Language detection indicators
                analysis['likely_language'] = 'english'  # Simplified assumption

                text_analysis[col] = analysis

                self.logger.info(f"Text analysis for {col}:")
                self.logger.info(f"  - Average length: {analysis['avg_length']:.1f} characters")
                self.logger.info(f"  - Average words: {analysis['avg_word_count']:.1f}")
                self.logger.info(f"  - Unique ratio: {analysis['unique_texts'] / analysis['total_texts']:.3f}")

            except Exception as e:
                self.logger.warning(f"Error analyzing text column {col}: {e}")
                continue

        # Store in feature store
        self.feature_store_data['text_analysis'] = text_analysis
        return text_analysis

    def ingest_uploaded_file(self, file: Union[BinaryIO, str], filename: str = None, mode: str = None) -> pd.DataFrame:
        """
        Load data from an uploaded file

        Args:
            file: File-like object or path to file
            filename: Original filename (if file is a file-like object)
            mode: Processing mode ('tabular', 'textual', 'mixed', or None for auto-detect)

        Returns:
            Pandas DataFrame containing the loaded data
        """
        section(f"LOADING UPLOADED FILE", self.logger)

        try:
            # Determine the filename
            if isinstance(file, str):
                # If file is a path
                file_path = file
                self.dataset_name = os.path.splitext(os.path.basename(file_path))[0]
            else:
                # If file is a file-like object
                if not filename:
                    # Generate a random name if none provided
                    self.dataset_name = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                else:
                    self.dataset_name = os.path.splitext(os.path.basename(filename))[0]

                # Read the file
                if hasattr(file, 'read'):
                    # If file is already a file-like object
                    self.logger.info(f"Reading uploaded file: {self.dataset_name}")
                    self.df = pd.read_csv(file)
                else:
                    self.logger.error("Invalid file object provided")
                    raise ValueError("Invalid file object provided")

            self.logger.info(f"Dataset name: {self.dataset_name}")

            # Set or detect processing mode
            if mode:
                self.set_processing_mode(mode)
            else:
                detected_mode = self.detect_processing_mode()
                self.set_processing_mode(detected_mode)

            # Create dataset-specific directories
            self.raw_data_dir = os.path.join(RAW_DATA_DIR, f"data_{self.dataset_name}")
            self.feature_store_dir = os.path.join(FEATURE_STORE_DIR, f"feature_store_{self.dataset_name}")
            self.plots_dir = os.path.join(REPORTS_DIR, f"plots_{self.dataset_name}")

            os.makedirs(self.raw_data_dir, exist_ok=True)
            os.makedirs(self.feature_store_dir, exist_ok=True)
            os.makedirs(self.plots_dir, exist_ok=True)

            # Add dataset name to feature store data
            self.feature_store_data['dataset_name'] = self.dataset_name
            self.feature_store_data['original_file_name'] = filename if filename else self.dataset_name

            # Save a copy of the original file
            raw_file_name = f"original.csv"
            raw_file_path = os.path.join(self.raw_data_dir, raw_file_name)

            self.logger.info(f"Saving raw data to: {raw_file_path}")
            self.df.to_csv(raw_file_path, index=False)

            self.logger.info(f"Successfully loaded CSV with shape: {self.df.shape}")
            self.logger.info(f"Processing mode: {self.processing_mode}")
            return self.df

        except Exception as e:
            self.logger.error(f"Failed to read uploaded file: {e}")
            raise

    def get_column_list(self) -> Dict[str, List[str]]:
        """
        Get list of columns from the loaded DataFrame, categorized by type

        Returns:
            Dictionary containing categorized column lists
        """
        if self.df is None:
            self.logger.error("No data loaded. Please load data first.")
            return {}

        # Get all columns
        all_columns = self.df.columns.tolist()

        # Identify different types
        textual_cols = self._identify_textual_columns()
        numerical_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = [col for col in self.df.select_dtypes(exclude=['number']).columns
                            if col not in textual_cols]

        return {
            'all_columns': all_columns,
            'textual_columns': textual_cols,
            'numerical_columns': numerical_cols,
            'categorical_columns': categorical_cols
        }

    def display_data_info(self) -> Dict:
        """
        Display basic information about the loaded data

        Returns:
            Dictionary containing basic data information
        """
        section("DATA PREVIEW AND INFORMATION", self.logger)

        if self.df is None:
            self.logger.error("No data loaded. Please load data first.")
            return {}

        # Get basic info
        self.logger.info(f"Data shape: {self.df.shape}")
        self.logger.info(f"Processing mode: {self.processing_mode}")
        self.logger.info(f"First 5 rows:\n{self.df.head()}")

        # Data types
        self.logger.info("Data types:")
        for col, dtype in self.df.dtypes.items():
            self.logger.info(f"  - {col}: {dtype}")

        # Missing values
        missing_values = self.df.isnull().sum()
        missing_percent = (missing_values / len(self.df)) * 100

        self.logger.info("Missing values:")
        columns_with_nulls = []
        for col, count in missing_values.items():
            if count > 0:
                columns_with_nulls.append(col)
                self.logger.warning(f"  - {col}: {count} missing values ({missing_percent[col]:.2f}%)")
            else:
                self.logger.info(f"  - {col}: No missing values")

        # Store columns with null values in feature store data
        self.feature_store_data['contains_null'] = columns_with_nulls

        # Basic statistics for numerical columns
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        if num_cols:
            self.logger.info("Numerical columns statistics:")
            stats_df = self.df[num_cols].describe().T
            for col in stats_df.index:
                stats_info = " | ".join([f"{stat}: {stats_df.loc[col, stat]:.2f}" for stat in stats_df.columns])
                self.logger.info(f"  - {col}: {stats_info}")

        # Return information as dictionary
        info_dict = {
            "shape": self.df.shape,
            "processing_mode": self.processing_mode,
            "columns": self.df.columns.tolist(),
            "dtypes": self.df.dtypes.to_dict(),
            "missing_values": missing_values.to_dict(),
            "missing_percent": missing_percent.to_dict()
        }

        return info_dict

    def identify_column_types(self) -> Tuple[List[str], List[str], List[str]]:
        section("IDENTIFYING COLUMN TYPES", self.logger)

        if self.df is None:
            self.logger.error("No data loaded. Please load data first.")
            return [], [], []

        # Store original columns
        self.feature_store_data['original_cols'] = self.df.columns.tolist()
        self.logger.info(f"Total columns: {len(self.feature_store_data['original_cols'])}")

        # First identify textual columns
        textual_cols = self._identify_textual_columns()

        # Then identify ID columns, excluding textual columns
        id_columns = self.identify_id_columns(exclude_cols=textual_cols)

        # Identify numerical and categorical columns, excluding ID and textual columns
        non_special_cols = [col for col in self.df.columns if col not in id_columns + textual_cols]
        numerical_cols = [col for col in self.df[non_special_cols].select_dtypes(include=['number']).columns]
        categorical_cols = [col for col in self.df[non_special_cols].select_dtypes(exclude=['number']).columns]

        # Update feature store
        self.feature_store_data['numerical_cols'] = numerical_cols
        self.feature_store_data['categorical_cols'] = categorical_cols
        self.feature_store_data['textual_cols'] = textual_cols
        self.feature_store_data['id_cols'] = id_columns  # Make sure this includes only non-textual IDs

        self.logger.info(f"Identified {len(numerical_cols)} numerical columns:")
        for col in numerical_cols:
            self.logger.info(f"  - {col}")

        self.logger.info(f"Identified {len(categorical_cols)} categorical columns:")
        for col in categorical_cols:
            self.logger.info(f"  - {col}")

        self.logger.info(f"Identified {len(textual_cols)} textual columns:")
        for col in textual_cols:
            self.logger.info(f"  - {col}")

        # Analyze text characteristics if textual columns found
        if textual_cols and self.processing_mode in ['textual', 'mixed']:
            self.analyze_text_characteristics()

        return numerical_cols, categorical_cols, textual_cols

    def identify_id_columns(self, exclude_cols: List[str] = None) -> List[str]:
        section("IDENTIFYING ID COLUMNS", self.logger)

        if self.df is None:
            self.logger.error("No data loaded. Please load data first.")
            return []

        id_columns = []
        exclude_cols = exclude_cols or []

        # Check all columns for ID-like characteristics
        for col in self.df.columns:
            if col in exclude_cols:
                continue

            # Skip columns with too many nulls
            null_percent = (self.df[col].isnull().sum() / len(self.df)) * 100
            if null_percent > 5:
                continue

            unique_ratio = self.df[col].nunique() / len(self.df)

            # Check if the column is numerical
            is_numerical = pd.api.types.is_numeric_dtype(self.df[col])

            # Check if the column name suggests it's an ID
            name_suggests_id = any(
                id_term in col.lower() for id_term in ['id', 'key', 'code', 'uuid', 'guid', 'Unnamed'])

            # Criteria for ID columns:
            if is_numerical:
                # For numerical columns: require both high unique ratio and name suggests ID
                is_id_column = (unique_ratio > ID_COLUMN_THRESHOLD) and name_suggests_id
            else:
                # For non-numerical: high uniqueness or moderate uniqueness with name suggesting ID
                is_id_column = (unique_ratio > ID_COLUMN_THRESHOLD) or (unique_ratio > 0.5 and name_suggests_id)

            if is_id_column:
                id_columns.append(col)
                self.logger.info(f"  - {col}: Identified as potential ID column (unique ratio: {unique_ratio:.4f})")

        self.feature_store_data['id_cols'] = id_columns
        self.logger.info(f"Identified {len(id_columns)} potential ID columns")

        return id_columns

    def analyze_distribution(self) -> Tuple[List[str], List[str]]:
        """
        Analyze column distributions to identify skewed and normally distributed columns

        Returns:
            Tuple containing lists of skewed and normally distributed column names
        """
        section("ANALYZING DISTRIBUTIONS", self.logger)

        if self.df is None:
            self.logger.error("No data loaded. Please load data first.")
            return [], []

        skewed_cols = []
        normal_cols = []

        # Skip ID columns when analyzing distributions
        id_cols = self.feature_store_data.get('id_cols', [])

        for col in self.feature_store_data['numerical_cols']:
            # Skip ID columns
            if col in id_cols:
                self.logger.info(f"Skipping {col} for distribution analysis (identified as ID column)")
                continue

            # Skip columns that are likely IDs or have too many unique values relative to row count
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio > 0.9:
                self.logger.info(
                    f"Skipping {col} for distribution analysis (likely ID column with {unique_ratio:.2f} unique ratio)")
                continue

            # Calculate skewness
            try:
                skewness = self.df[col].skew()

                # Perform Shapiro-Wilk test for normality (on a sample if dataset is large)
                sample = self.df[col].dropna()
                if len(sample) > 5000:  # Sample for large datasets
                    sample = sample.sample(5000, random_state=RANDOM_STATE)

                shapiro_test = stats.shapiro(sample)
                p_value = shapiro_test.pvalue

                # Use primarily skewness for classification, but report p-value for reference
                # Only consider a column skewed if its absolute skewness exceeds the threshold
                if abs(skewness) > SKEWNESS_THRESHOLD:
                    skewed_cols.append(col)
                    classification = "Skewed"
                else:
                    normal_cols.append(col)
                    classification = "Normal"

                self.logger.info(
                    f"  - {col}: {classification} distribution (skewness={skewness:.4f}, p-value={p_value:.4f})")

            except Exception as e:
                self.logger.warning(f"Error analyzing distribution for {col}: {e}")

        self.feature_store_data['skewed_cols'] = skewed_cols
        self.feature_store_data['normal_cols'] = normal_cols

        self.logger.info(
            f"Identified {len(skewed_cols)} skewed columns and {len(normal_cols)} normally distributed columns")

        return skewed_cols, normal_cols

    def detect_outliers(self) -> List[str]:
        """
        Detect outliers in numerical columns using IQR method

        Returns:
            List of column names containing outliers
        """
        section("DETECTING OUTLIERS", self.logger)

        if self.df is None:
            self.logger.error("No data loaded. Please load data first.")
            return []

        columns_with_outliers = []

        # Skip ID columns when detecting outliers
        id_cols = self.feature_store_data.get('id_cols', [])

        for col in self.feature_store_data['numerical_cols']:
            # Skip ID columns
            if col in id_cols:
                self.logger.info(f"Skipping {col} for outlier detection (identified as ID column)")
                continue

            # Skip columns that are likely IDs or have too many unique values
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio > 0.9:
                continue

            # Calculate IQR
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier boundaries
            lower_bound = Q1 - OUTLIER_THRESHOLD * IQR
            upper_bound = Q3 + OUTLIER_THRESHOLD * IQR

            # Count outliers
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percent = (outlier_count / len(self.df)) * 100

            if outlier_count > 0:
                columns_with_outliers.append(col)
                self.logger.info(f"  - {col}: {outlier_count} outliers ({outlier_percent:.2f}%)")

        self.feature_store_data['contains_outliers'] = columns_with_outliers
        self.logger.info(f"Identified {len(columns_with_outliers)} columns with outliers")

        return columns_with_outliers

    def analyze_correlations(self) -> Dict[str, List[str]]:
        """
        Analyze correlations between numerical columns

        Returns:
            Dictionary mapping column names to lists of highly correlated columns
        """
        section("ANALYZING CORRELATIONS", self.logger)

        if self.df is None:
            self.logger.error("No data loaded. Please load data first.")
            return {}

        # Skip ID columns when analyzing correlations
        id_cols = self.feature_store_data.get('id_cols', [])
        numerical_cols = [col for col in self.feature_store_data['numerical_cols'] if col not in id_cols]

        # Calculate correlation matrix for numerical columns
        try:
            if len(numerical_cols) < 2:
                self.logger.warning("Not enough non-ID numerical columns to calculate correlations")
                return {}

            corr_matrix = self.df[numerical_cols].corr()

            correlated_cols = {}

            # Find highly correlated features
            for i, col1 in enumerate(corr_matrix.columns):
                correlated_features = []

                for col2 in corr_matrix.columns:
                    if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > CORRELATION_THRESHOLD:
                        # Convert NumPy scalar to Python float for proper YAML serialization
                        correlation_value = float(corr_matrix.loc[col1, col2])

                        correlated_features.append({
                            'column': col2,
                            'correlation': correlation_value  # Now it's a Python float
                        })

                if correlated_features:
                    correlated_cols[col1] = correlated_features
                    self.logger.info(f"Column {col1} is highly correlated with:")
                    for item in correlated_features:
                        self.logger.info(f"  - {item['column']} (correlation: {item['correlation']:.4f})")

            self.feature_store_data['correlated_cols'] = correlated_cols
            return correlated_cols

        except Exception as e:
            self.logger.error(f"Failed to analyze correlations: {e}")
            return {}

    def set_target_column(self, target_col: str) -> str:
        """
        Set the target column for processing

        Args:
            target_col: Name of the target column

        Returns:
            Name of the selected target column or None if invalid
        """
        section("TARGET COLUMN SELECTION", self.logger)

        if self.df is None:
            self.logger.error("No data loaded. Please load data first.")
            return None

        if target_col in self.df.columns:
            self.feature_store_data['target_col'] = target_col
            self.logger.info(f"Target column set to: {target_col}")
            return target_col
        else:
            self.logger.error(f"Target column '{target_col}' not found in dataset columns.")
            return None

    def save_feature_store_yaml(self) -> str:
        section("SAVING FEATURE STORE METADATA", self.logger)

        if not self.feature_store_data['target_col']:
            self.logger.warning("Target column not set. Feature store will be incomplete.")

        try:
            # Custom YAML representer for NumPy data types
            def numpy_representer(dumper, data):
                if isinstance(data, (np.integer, np.floating)):
                    return dumper.represent_scalar('tag:yaml.org,2002:float', float(data))
                elif isinstance(data, np.ndarray):
                    return dumper.represent_sequence('tag:yaml.org,2002:seq', data.tolist())
                return None

            # Register the representer for numpy types
            yaml.add_representer(np.int64, numpy_representer)
            yaml.add_representer(np.int32, numpy_representer)
            yaml.add_representer(np.float64, numpy_representer)
            yaml.add_representer(np.float32, numpy_representer)
            yaml.add_representer(np.ndarray, numpy_representer)

            # Convert any remaining numpy types to native Python types
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            feature_store_data_converted = convert_numpy_types(self.feature_store_data)

            yaml_path = os.path.join(self.feature_store_dir, 'feature_store.yaml')
            with open(yaml_path, 'w') as f:
                yaml.dump(feature_store_data_converted, f, default_flow_style=False, sort_keys=False)

            self.logger.info(f"Feature store metadata saved to: {yaml_path}")
            return yaml_path

        except Exception as e:
            self.logger.error(f"Failed to save feature store YAML: {e}")
            return None

    def save_intel_yaml(self) -> str:
        """
        Save dataset information to intel.yaml in the main project directory

        Returns:
            Path to the saved YAML file
        """
        section("SAVING INTEL YAML", self.logger)

        try:
            intel_data = {
                'dataset_name': self.dataset_name,
                'original_file_name': self.feature_store_data.get('original_file_name', self.dataset_name),
                'processing_mode': self.processing_mode,  # New: processing mode
                'processed_timestamp': datetime.now().strftime(DATETIME_FORMAT),
                'feature_store_path': os.path.join(self.feature_store_dir, 'feature_store.yaml'),
                'train_path': os.path.join(self.raw_data_dir, 'train.csv'),
                'test_path': os.path.join(self.raw_data_dir, 'test.csv'),
                'plots_dir': self.plots_dir,
                'target_column': self.feature_store_data['target_col']
            }

            yaml_path = os.path.join(self.project_dir, 'intel.yaml')
            with open(yaml_path, 'w') as f:
                yaml.dump(intel_data, f, default_flow_style=False, sort_keys=False)

            self.logger.info(f"Intel metadata saved to: {yaml_path}")
            return yaml_path

        except Exception as e:
            self.logger.error(f"Failed to save intel YAML: {e}")
            return None

    def analyze_text_characteristics(self) -> Dict[str, Dict]:
        section("ANALYZING TEXT CHARACTERISTICS", self.logger)

        text_analysis = {}
        textual_cols = self.feature_store_data.get('textual_cols', [])

        for col in textual_cols:
            try:
                # Convert to string and drop nulls
                text_series = self.df[col].dropna().astype(str)

                if len(text_series) == 0:
                    self.logger.warning(f"Skipping text analysis for {col}: no valid text data")
                    continue

                # Basic statistics
                analysis = {
                    'total_texts': len(text_series),
                    'unique_texts': text_series.nunique(),
                    'avg_length': float(text_series.str.len().mean()),
                    'max_length': int(text_series.str.len().max()),
                    'min_length': int(text_series.str.len().min()),
                    'avg_word_count': float(text_series.str.split().str.len().mean()),
                    'contains_punctuation': float(text_series.str.contains(r'[.!?]').mean()),
                    'avg_sentence_count': float(text_series.str.split(r'[.!?]').str.len().mean()),
                    'likely_language': 'english'  # Simplified assumption
                }

                text_analysis[col] = analysis

                self.logger.info(f"Text analysis for {col}:")
                self.logger.info(f"  - Average length: {analysis['avg_length']:.1f} characters")
                self.logger.info(f"  - Average words: {analysis['avg_word_count']:.1f}")
                self.logger.info(f"  - Unique ratio: {analysis['unique_texts'] / analysis['total_texts']:.3f}")

            except Exception as e:
                self.logger.warning(f"Error analyzing text column {col}: {e}")
                continue

        # Store in feature store
        self.feature_store_data['text_analysis'] = text_analysis
        return text_analysis

    def generate_data_profile(self) -> Dict[str, str]:
        """
        Generate and save basic data profile plots

        Returns:
            Dictionary with plot paths
        """
        section("GENERATING DATA PROFILE", self.logger)

        if self.df is None:
            self.logger.error("No data loaded. Please load data first.")
            return {}

        try:
            plot_paths = {}

            # Skip ID columns when generating plots
            id_cols = self.feature_store_data.get('id_cols', [])
            plot_columns = [col for col in self.feature_store_data['numerical_cols'] if col not in id_cols][:10]

            # Plot distributions for numerical columns (only for tabular/mixed modes)
            if self.processing_mode in ['tabular', 'mixed'] and plot_columns:
                self.logger.info("Generating distribution plots for numerical columns")
                for col in plot_columns:  # Limit to 10 columns, excluding IDs
                    plt.figure(figsize=(10, 4))

                    # Histogram with KDE
                    plt.subplot(1, 2, 1)
                    sns.histplot(self.df[col], kde=True)
                    plt.title(f'Distribution of {col}')

                    # Box plot
                    plt.subplot(1, 2, 2)
                    sns.boxplot(x=self.df[col])
                    plt.title(f'Boxplot of {col}')

                    # Save plot
                    plot_path = os.path.join(self.plots_dir, f'distribution_{col}.png')
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    plt.close()

                    plot_paths[f'distribution_{col}'] = plot_path
                    self.logger.info(f"Saved distribution plot for {col} to {plot_path}")

            # Plot correlation heatmap (excluding ID columns) - only for tabular/mixed modes
            if self.processing_mode in ['tabular', 'mixed']:
                self.logger.info("Generating correlation heatmap")
                numerical_cols = [col for col in self.feature_store_data['numerical_cols'] if col not in id_cols]

                if len(numerical_cols) > 1:  # Need at least 2 columns for correlation
                    corr_matrix = self.df[numerical_cols].corr()

                    plt.figure(figsize=(12, 10))
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                                center=0, square=True, linewidths=.5)
                    plt.title('Correlation Heatmap')

                    # Save correlation heatmap
                    heatmap_path = os.path.join(self.plots_dir, 'correlation_heatmap.png')
                    plt.tight_layout()
                    plt.savefig(heatmap_path)
                    plt.close()

                    plot_paths['correlation_heatmap'] = heatmap_path
                    self.logger.info(f"Saved correlation heatmap to {heatmap_path}")

            # Generate text-specific visualizations for textual/mixed modes
            if self.processing_mode in ['textual', 'mixed']:
                textual_cols = self.feature_store_data.get('textual_cols', [])
                if textual_cols:
                    self._generate_text_visualizations(textual_cols, plot_paths)

            return plot_paths

        except Exception as e:
            self.logger.error(f"Failed to generate data profile: {e}")
            return {}

    def _generate_text_visualizations(self, textual_cols: List[str], plot_paths: Dict[str, str]):
        """
        Generate visualizations specific to textual data

        Args:
            textual_cols: List of textual column names
            plot_paths: Dictionary to store plot paths
        """
        self.logger.info("Generating text-specific visualizations")

        for col in textual_cols[:5]:  # Limit to 5 text columns
            try:
                # Text length distribution
                text_series = self.df[col].dropna().astype(str)
                text_lengths = text_series.str.len()

                plt.figure(figsize=(12, 4))

                # Text length histogram
                plt.subplot(1, 2, 1)
                plt.hist(text_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title(f'Text Length Distribution - {col}')
                plt.xlabel('Text Length (characters)')
                plt.ylabel('Frequency')

                # Word count distribution
                word_counts = text_series.str.split().str.len()
                plt.subplot(1, 2, 2)
                plt.hist(word_counts, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
                plt.title(f'Word Count Distribution - {col}')
                plt.xlabel('Word Count')
                plt.ylabel('Frequency')

                # Save plot
                plot_path = os.path.join(self.plots_dir, f'text_analysis_{col}.png')
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()

                plot_paths[f'text_analysis_{col}'] = plot_path
                self.logger.info(f"Saved text analysis plot for {col} to {plot_path}")

            except Exception as e:
                self.logger.warning(f"Error generating text visualization for {col}: {e}")
                continue

    def perform_train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets

        Returns:
            Tuple of (train_df, test_df)
        """
        section("PERFORMING TRAIN-TEST SPLIT", self.logger)

        if self.df is None:
            self.logger.error("No data loaded. Please load data first.")
            return None, None

        try:
            # Check if target column is set
            if self.feature_store_data['target_col'] is None:
                self.logger.warning("Target column not set. Using random split.")
                train_df, test_df = train_test_split(
                    self.df,
                    test_size=TEST_SIZE,
                    random_state=RANDOM_STATE
                )
            else:
                # Stratified split if target is categorical with limited unique values
                target_col = self.feature_store_data['target_col']
                if target_col in self.feature_store_data['categorical_cols'] or self.df[target_col].nunique() < 10:
                    self.logger.info(f"Using stratified split on target column: {target_col}")
                    train_df, test_df = train_test_split(
                        self.df,
                        test_size=TEST_SIZE,
                        random_state=RANDOM_STATE,
                        stratify=self.df[target_col]
                    )
                else:
                    self.logger.info(f"Using random split (target column {target_col} not suitable for stratification)")
                    train_df, test_df = train_test_split(
                        self.df,
                        test_size=TEST_SIZE,
                        random_state=RANDOM_STATE
                    )

            # Save train and test datasets
            train_path = os.path.join(self.raw_data_dir, 'train.csv')
            test_path = os.path.join(self.raw_data_dir, 'test.csv')

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            self.logger.info(f"Train set saved to {train_path} with shape {train_df.shape}")
            self.logger.info(f"Test set saved to {test_path} with shape {test_df.shape}")

            # Add split info to feature store data
            self.feature_store_data['train_rows'] = train_df.shape[0]
            self.feature_store_data['test_rows'] = test_df.shape[0]

            return train_df, test_df

        except Exception as e:
            self.logger.error(f"Failed to perform train-test split: {e}")
            return None, None

    def run_ingestion_pipeline(self, file: Union[BinaryIO, str], filename: str = None, target_col: str = None,
                               mode: str = None) -> Dict:
        """
        Run the complete data ingestion pipeline

        Args:
            file: File-like object or path to the CSV file
            filename: Original filename (if file is file-like object)
            target_col: Name of the target column (optional)
            mode: Processing mode ('tabular', 'textual', 'mixed', or None for auto-detect)

        Returns:
            Dictionary containing feature store information
        """
        section("STARTING DATA INGESTION PIPELINE", self.logger, char='*', length=80)

        try:
            # Step 1: Load the file with mode detection/setting
            self.ingest_uploaded_file(file, filename, mode)

            # Step 2: Display basic information
            data_info = self.display_data_info()

            # Step 3: Identify column types (including ID and textual columns)
            numerical_cols, categorical_cols, textual_cols = self.identify_column_types()

            # Step 4: Analyze distributions (only for numerical columns in tabular/mixed mode)
            if self.processing_mode in ['tabular', 'mixed']:
                skewed_cols, normal_cols = self.analyze_distribution()
                # Step 5: Detect outliers (only for tabular/mixed mode)
                outlier_cols = self.detect_outliers()
                # Step 6: Analyze correlations (only for tabular/mixed mode)
                correlated_cols = self.analyze_correlations()
            else:
                skewed_cols, normal_cols, outlier_cols, correlated_cols = [], [], [], {}

            # Step 7: Set target column if provided
            if target_col:
                self.set_target_column(target_col)

            # Step 8: Perform train-test split
            train_df, test_df = self.perform_train_test_split()

            # Step 9: Generate basic profile plots
            plot_paths = self.generate_data_profile()

            # Step 10: Save feature store YAML
            feature_store_path = self.save_feature_store_yaml()

            # Step 11: Save intel YAML
            intel_path = self.save_intel_yaml()

            section("DATA INGESTION PIPELINE COMPLETED SUCCESSFULLY", self.logger, char='*', length=80)

            # Create a results dictionary for API response
            results = {
                "dataset_name": self.dataset_name,
                "processing_mode": self.processing_mode,
                "data_shape": self.df.shape,
                "numerical_columns": numerical_cols,
                "categorical_columns": categorical_cols,
                "textual_columns": textual_cols,
                "skewed_columns": skewed_cols if self.processing_mode in ['tabular', 'mixed'] else [],
                "columns_with_outliers": outlier_cols if self.processing_mode in ['tabular', 'mixed'] else [],
                "feature_store_path": feature_store_path,
                "intel_path": intel_path,
                "plots_dir": self.plots_dir,
                "train_path": os.path.join(self.raw_data_dir, 'train.csv'),
                "test_path": os.path.join(self.raw_data_dir, 'test.csv'),
                "feature_store_data": self.feature_store_data
            }

            return results

        except Exception as e:
            self.logger.error(f"Data ingestion pipeline failed: {e}")
            section("DATA INGESTION PIPELINE FAILED", self.logger, char='*', length=80)
            raise


# Function to create a DataIngestion instance
def create_data_ingestion(project_dir=None):
    """Factory function to create and configure a DataIngestion instance"""
    # Configure logger
    configure_logger()

    # Create DataIngestion instance
    return DataIngestion(project_dir)


# This allows the file to be imported without automatically running anything
if __name__ == "__main__":
    # This code runs only when the file is executed directly
    # but not when imported as a module
    print("This module is designed to be imported by a FastAPI application.")
    print("Run your FastAPI app instead of this file directly.")