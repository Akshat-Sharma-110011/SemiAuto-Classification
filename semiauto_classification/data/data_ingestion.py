import os
import sys
import traceback

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
import zipfile
from pathlib import Path
from PIL import Image

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
        # Configure logger with UTF-8 encoding for Windows compatibility
        import sys
        import codecs

        # Set stdout to use UTF-8 encoding
        if sys.platform.startswith('win'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
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

# Image processing constants
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
MAX_IMAGE_SIZE_MB = 50  # Maximum size for individual image files
IMAGE_ANALYSIS_SAMPLE_SIZE = 100  # Number of images to sample for analysis


class DataIngestion:
    """
    Class for handling data ingestion from uploaded files, performing initial analysis,
    and creating feature store metadata.

    Supports tabular, textual, and image data processing modes.
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

        # Image-specific attributes
        self.images_dir = None
        self.labels_dir = None

        self.feature_store_data = {
            'original_cols': [],
            'numerical_cols': [],
            'categorical_cols': [],
            'textual_cols': [],  # New: textual columns
            'image_cols': [],  # New: image columns
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
            'test_size': TEST_SIZE,
            'images_path': None,  # New: path to images directory
            'labels_path': None,  # New: path to labels directory
        }

    def detect_processing_mode(self) -> str:
        """
        Automatically detect the processing mode based on data characteristics

        Returns:
            str: 'tabular', 'textual', 'image', or 'mixed'
        """
        section("DETECTING PROCESSING MODE", self.logger)

        # If image directories are provided, it's image mode
        if self.images_dir and self.labels_dir:
            self.logger.info("Images and labels directories provided - using image mode")
            return "image"

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
            mode: 'tabular', 'textual', 'image', or 'mixed'

        Returns:
            str: The set processing mode
        """
        valid_modes = ['tabular', 'textual', 'image', 'mixed']
        if mode not in valid_modes:
            self.logger.warning(f"Invalid mode '{mode}'. Using 'tabular' as default.")
            mode = 'tabular'

        self.processing_mode = mode
        self.feature_store_data['processing_mode'] = mode
        self.logger.info(f"Processing mode set to: {mode}")
        return mode

    def _is_image_file(self, file_path: str) -> bool:
        """
        Check if a file is a supported image format

        Args:
            file_path: Path to the file

        Returns:
            bool: True if file is a supported image format
        """
        return Path(file_path).suffix.lower() in SUPPORTED_IMAGE_FORMATS

    def _analyze_image_dataset(self, images_dir: str, labels_csv_path: str) -> Dict:
        """
        Analyze image dataset structure and characteristics.
        """
        section("ANALYZING IMAGE DATASET", self.logger)

        images_path = Path(images_dir)
        labels_csv = Path(labels_csv_path)

        self.logger.info(f"Analyzing images directory: {images_path}")
        self.logger.info(f"Analyzing labels CSV: {labels_csv}")
        self.logger.info(f"Images directory exists: {images_path.exists()}")
        self.logger.info(f"Labels CSV exists: {labels_csv.exists()}")

        analysis = {
            "total_images": 0,
            "total_labels": 0,
            "image_formats": {},
            "image_sizes": [],
            "label_format": "csv",
            "classes": [],
            "class_distribution": {},
            "avg_image_size": None,
            "total_dataset_size_mb": 0,
            "image_column": None,
            "target_column": None,
        }

        # ---------------- Images ----------------
        if images_path.exists():
            self.logger.info("Starting image directory analysis...")

            all_files = list(images_path.rglob("*"))
            self.logger.info(f"Found {len(all_files)} total files in images directory")

            image_files = [f for f in all_files if f.is_file() and self._is_image_file(str(f))]
            analysis["total_images"] = len(image_files)
            self.logger.info(f"Found {analysis['total_images']} image files")
            self.logger.info(f"Supported image formats: {SUPPORTED_IMAGE_FORMATS}")

            if image_files:
                for i, img_file in enumerate(image_files[:5]):
                    self.logger.info(f"Sample image {i + 1}: {img_file.name}")

            sample_size = min(IMAGE_ANALYSIS_SAMPLE_SIZE, len(image_files))
            sample_images = image_files[:sample_size]
            self.logger.info(f"Analyzing sample of {sample_size} images")

            for i, img_path in enumerate(sample_images):
                try:
                    ext = img_path.suffix.lower()
                    analysis["image_formats"][ext] = analysis["image_formats"].get(ext, 0) + 1
                    analysis["total_dataset_size_mb"] += img_path.stat().st_size / (1024 * 1024)
                    with Image.open(img_path) as img:
                        analysis["image_sizes"].append(img.size)
                    if i < 3:
                        self.logger.info(
                            f"Analyzed image {img_path.name}: format={ext}, size={img.size}"
                        )
                except Exception as e:
                    self.logger.warning(f"Could not analyze image {img_path}: {e}")

            if analysis["image_sizes"]:
                avg_w = sum(w for w, _ in analysis["image_sizes"]) / len(analysis["image_sizes"])
                avg_h = sum(h for _, h in analysis["image_sizes"]) / len(analysis["image_sizes"])
                analysis["avg_image_size"] = (int(avg_w), int(avg_h))
                self.logger.info(f"Average image size: {analysis['avg_image_size']}")
            else:
                self.logger.warning("No image sizes collected.")
        else:
            self.logger.error(f"Images directory does not exist: {images_path}")

        # ---------------- Labels ----------------
        target_column = None
        if labels_csv.exists():
            self.logger.info("Starting labels CSV analysis...")
            try:
                df = pd.read_csv(labels_csv)
                analysis["total_labels"] = len(df)

                self.logger.info(f"Labels CSV shape: {df.shape}")
                self.logger.info(f"Columns: {list(df.columns)}")
                for i, (_, row) in enumerate(df.head(3).iterrows()):
                    self.logger.info(f"Row {i}: {dict(row)}")

                image_column = self._detect_image_column(df)
                analysis["image_column"] = image_column
                self.logger.info(f"Detected image column: {image_column}")

                if image_column:
                    sample_image_values = df[image_column].head(5).tolist()
                    self.logger.info(f"Sample values from image column: {sample_image_values}")

                    possible_targets = [c for c in df.columns if c != image_column]
                    if len(possible_targets) == 1:
                        target_column = possible_targets[0]
                        self.logger.info(f"Single target column: {target_column}")
                    elif len(possible_targets) > 1:
                        self.logger.info("Multiple target columns, analyzing uniqueness...")
                        for col in possible_targets:
                            unique_ratio = df[col].nunique() / len(df)
                            self.logger.info(f"{col}: unique_ratio={unique_ratio:.3f}")
                            if unique_ratio < 0.5:
                                target_column = col
                                self.logger.info(f"Using {col} as target.")
                                break
                        if not target_column:
                            target_column = possible_targets[0]
                            self.logger.info(f"No clear target; defaulting to {target_column}")
                    else:
                        self.logger.warning("No possible target columns found")

                analysis["target_column"] = target_column

                if target_column and target_column in df.columns:
                    analysis["classes"] = df[target_column].unique().tolist()
                    analysis["class_distribution"] = df[target_column].value_counts().to_dict()
                    max_c = max(analysis["class_distribution"].values())
                    min_c = min(analysis["class_distribution"].values())
                    if min_c > 0 and max_c / min_c > 10:
                        self.logger.warning(
                            f"Significant class imbalance (ratio={max_c / min_c:.2f})"
                        )
            except Exception as e:
                self.logger.error(f"Failed to analyze labels CSV: {e}")
                self.logger.error(traceback.format_exc())
        else:
            self.logger.error(f"Labels CSV does not exist: {labels_csv}")

        # ----------- Final report -----------
        self.logger.info("=== IMAGE DATASET ANALYSIS RESULTS ===")
        self.logger.info(f"Total images: {analysis['total_images']}")
        self.logger.info(f"Total labels: {analysis['total_labels']}")
        self.logger.info(f"Image formats: {analysis['image_formats']}")
        self.logger.info(f"Classes found: {len(analysis['classes'])}")
        self.logger.info(f"Average image size: {analysis['avg_image_size']}")
        self.logger.info(f"Dataset size: {analysis['total_dataset_size_mb']:.2f} MB")
        self.logger.info(f"Image column: {analysis['image_column']}")
        self.logger.info(f"Target column: {analysis['target_column']}")
        self.logger.info("========================================")

        return analysis

    def _detect_image_column(self, df: pd.DataFrame) -> str:
        """
        Detect which column contains image file names or paths

        Args:
            df: DataFrame containing labels

        Returns:
            Name of the column containing image references
        """
        self.logger.info("Starting image column detection...")

        for col in df.columns:
            self.logger.info(f"Checking column: {col}")

            # Check if column contains file extensions
            sample_values = df[col].dropna().astype(str).head(10)
            self.logger.info(f"Sample values from {col}: {sample_values.tolist()}")

            has_image_extensions = any(
                any(ext in str(val).lower() for ext in SUPPORTED_IMAGE_FORMATS)
                for val in sample_values
            )
            self.logger.info(f"Column {col} has image extensions: {has_image_extensions}")

            # Check for common image column names
            has_image_name = any(keyword in col.lower() for keyword in ['image', 'file', 'path', 'name', 'filename'])
            self.logger.info(f"Column {col} has image-related name: {has_image_name}")

            if has_image_extensions or has_image_name:
                self.logger.info(f"Selected {col} as image column")
                return col

        # Default to first column if no clear image column found
        default_col = df.columns[0]
        self.logger.warning(f"No clear image column found, defaulting to first column: {default_col}")
        return default_col

    def ingest_image_dataset(self, images_zip: Union[BinaryIO, str], labels_csv: Union[BinaryIO, str],
                             dataset_name: str = None) -> Dict:
        """
        Load image dataset from uploaded zip file and labels CSV

        Args:
            images_zip: Zip file containing images
            labels_csv: CSV file containing labels
            dataset_name: Name for the dataset

        Returns:
            Dictionary containing dataset information
        """
        section("LOADING IMAGE DATASET", self.logger)

        try:
            # Set dataset name
            if not dataset_name:
                self.dataset_name = f"image_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                self.dataset_name = dataset_name

            self.logger.info(f"Dataset name: {self.dataset_name}")
            self.logger.info(f"Images zip type: {type(images_zip)}")
            self.logger.info(f"Labels CSV type: {type(labels_csv)}")

            # Create dataset-specific directories
            self.raw_data_dir = os.path.join(RAW_DATA_DIR, f"data_{self.dataset_name}")
            self.feature_store_dir = os.path.join(FEATURE_STORE_DIR, f"feature_store_{self.dataset_name}")
            self.plots_dir = os.path.join(REPORTS_DIR, f"plots_{self.dataset_name}")

            self.logger.info(f"Creating directories:")
            self.logger.info(f"  - Raw data dir: {self.raw_data_dir}")
            self.logger.info(f"  - Feature store dir: {self.feature_store_dir}")
            self.logger.info(f"  - Plots dir: {self.plots_dir}")

            os.makedirs(self.raw_data_dir, exist_ok=True)
            os.makedirs(self.feature_store_dir, exist_ok=True)
            os.makedirs(self.plots_dir, exist_ok=True)

            # Create directory structure with corrected paths
            original_dir = os.path.join(self.raw_data_dir, 'original')
            original_labels_dir = os.path.join(original_dir, 'labels')
            train_dir = os.path.join(self.raw_data_dir, 'train')
            test_dir = os.path.join(self.raw_data_dir, 'test')
            train_images_dir = os.path.join(train_dir, 'images')
            test_images_dir = os.path.join(test_dir, 'images')
            train_labels_dir = os.path.join(train_dir, 'labels')
            test_labels_dir = os.path.join(test_dir, 'labels')

            directories_to_create = [
                original_dir, original_labels_dir, train_dir, test_dir,
                train_images_dir, test_images_dir, train_labels_dir, test_labels_dir
            ]

            self.logger.info(f"Creating {len(directories_to_create)} subdirectories:")
            for directory in directories_to_create:
                os.makedirs(directory, exist_ok=True)
                self.logger.info(f"  - Created: {directory}")

            # Extract images to original directory
            self.logger.info("Extracting images from zip file...")
            try:
                if isinstance(images_zip, str):
                    self.logger.info(f"Images zip is file path: {images_zip}")
                    with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                        zip_contents = zip_ref.namelist()
                        self.logger.info(f"Zip contains {len(zip_contents)} files")
                        self.logger.info(f"First 5 files in zip: {zip_contents[:5]}")
                        zip_ref.extractall(original_dir)
                        self.logger.info(f"Successfully extracted zip to: {original_dir}")
                else:
                    self.logger.info("Images zip is file-like object")
                    with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                        zip_contents = zip_ref.namelist()
                        self.logger.info(f"Zip contains {len(zip_contents)} files")
                        self.logger.info(f"First 5 files in zip: {zip_contents[:5]}")
                        zip_ref.extractall(original_dir)
                        self.logger.info(f"Successfully extracted zip to: {original_dir}")
            except Exception as e:
                self.logger.error(f"Failed to extract images zip: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                raise

            # Find the images directory (created by unzipping)
            self.logger.info("Locating extracted images directory...")
            extracted_contents = os.listdir(original_dir)
            self.logger.info(f"Contents of original directory: {extracted_contents}")

            images_source_dir = None

            # Look for the images directory or the first directory that contains images
            for item in extracted_contents:
                item_path = os.path.join(original_dir, item)
                self.logger.info(f"Checking item: {item} (is_dir: {os.path.isdir(item_path)})")

                if os.path.isdir(item_path) and item.lower() != 'labels':
                    # Check if this directory contains images
                    try:
                        dir_contents = os.listdir(item_path)
                        self.logger.info(f"Contents of {item}: {dir_contents[:10]}...")

                        image_files = [f for f in dir_contents
                                       if any(f.lower().endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS)]
                        self.logger.info(f"Found {len(image_files)} image files in {item}")

                        if image_files:
                            images_source_dir = item_path
                            self.logger.info(f"Selected {item_path} as images source directory")
                            self.logger.info(f"Sample image files: {image_files[:5]}")
                            break
                    except Exception as e:
                        self.logger.warning(f"Could not check contents of {item_path}: {e}")

            if not images_source_dir:
                # If no subdirectory found, images might be directly in original_dir
                self.logger.info("No subdirectory with images found, checking original directory...")
                try:
                    original_contents = os.listdir(original_dir)
                    image_files = [f for f in original_contents
                                   if any(f.lower().endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS)]
                    self.logger.info(f"Found {len(image_files)} image files directly in original directory")

                    if image_files:
                        images_source_dir = original_dir
                        self.logger.info(f"Using original directory as images source: {original_dir}")
                    else:
                        raise ValueError("No images found in the uploaded zip file")
                except Exception as e:
                    self.logger.error(f"Error checking original directory: {e}")
                    raise ValueError("No images found in the uploaded zip file")

            self.logger.info(f"Final images source directory: {images_source_dir}")

            # Save labels CSV to original/labels/labels.csv
            labels_csv_path = os.path.join(original_labels_dir, 'labels.csv')
            self.logger.info(f"Saving labels CSV to: {labels_csv_path}")

            try:
                if isinstance(labels_csv, str):
                    self.logger.info(f"Labels CSV is file path: {labels_csv}")
                    shutil.copy(labels_csv, labels_csv_path)
                    self.logger.info("Successfully copied labels CSV file")
                else:
                    self.logger.info("Labels CSV is file-like object")
                    labels_content = labels_csv.read()
                    if isinstance(labels_content, bytes):
                        labels_content = labels_content.decode('utf-8')
                        self.logger.info("Decoded labels content from bytes to string")

                    with open(labels_csv_path, 'w', encoding='utf-8') as f:
                        f.write(labels_content)
                    self.logger.info("Successfully wrote labels CSV content to file")

                # Verify the saved CSV
                test_df = pd.read_csv(labels_csv_path)
                self.logger.info(f"Verified saved CSV: shape={test_df.shape}, columns={list(test_df.columns)}")

            except Exception as e:
                self.logger.error(f"Failed to save labels CSV: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                raise

            # Set processing mode to image
            self.logger.info("Setting processing mode to 'image'")
            self.set_processing_mode('image')

            # Analyze the image dataset
            self.logger.info("Starting image dataset analysis...")
            try:
                image_analysis = self._analyze_image_dataset(images_source_dir, labels_csv_path)
                self.logger.info("Image dataset analysis completed successfully")
            except Exception as e:
                self.logger.error(f"Failed to analyze image dataset: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                raise

            # Update feature store data
            self.logger.info("Updating feature store data...")
            self.feature_store_data.update({
                'dataset_name': self.dataset_name,
                'original_images_path': images_source_dir,
                'original_labels_path': labels_csv_path,
                'train_images_path': train_images_dir,
                'test_images_path': test_images_dir,
                'train_labels_path': os.path.join(train_labels_dir, 'labels.csv'),
                'test_labels_path': os.path.join(test_labels_dir, 'labels.csv'),
                'image_analysis': image_analysis,
                'processing_mode': 'image',
                'image_column': image_analysis.get('image_column'),
                'target_column': image_analysis.get('target_column')
            })

            self.logger.info("=== IMAGE DATASET INGESTION SUMMARY ===")
            self.logger.info(f"Dataset name: {self.dataset_name}")
            self.logger.info(f"Original images path: {images_source_dir}")
            self.logger.info(f"Original labels path: {labels_csv_path}")
            self.logger.info(f"Image column: {image_analysis.get('image_column')}")
            self.logger.info(f"Target column: {image_analysis.get('target_column')}")
            self.logger.info(f"Total images: {image_analysis.get('total_images', 0)}")
            self.logger.info(f"Total labels: {image_analysis.get('total_labels', 0)}")
            self.logger.info(f"Classes: {image_analysis.get('classes', [])}")
            self.logger.info("========================================")

            return {
                'dataset_name': self.dataset_name,
                'processing_mode': 'image',
                'original_images_path': images_source_dir,
                'original_labels_path': labels_csv_path,
                'image_analysis': image_analysis
            }

        except Exception as e:
            self.logger.error(f"Failed to load image dataset: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

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
            mode: Processing mode ('tabular', 'textual', 'image', 'mixed', or None for auto-detect)

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
        if self.processing_mode == 'image':
            return {
                'all_columns': [],
                'textual_columns': [],
                'numerical_columns': [],
                'categorical_columns': [],
                'image_columns': ['images', 'labels']
            }

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
            'categorical_columns': categorical_cols,
            'image_columns': []
        }

    def display_data_info(self) -> Dict:
        """
        Display basic information about the loaded data

        Returns:
            Dictionary containing basic data information
        """
        section("DATA PREVIEW AND INFORMATION", self.logger)

        # Handle image mode differently
        if self.processing_mode == 'image':
            if not (self.images_dir and self.labels_dir):
                self.logger.error("No image data loaded. Please load image dataset first.")
                return {}

            image_analysis = self.feature_store_data.get('image_analysis', {})

            self.logger.info(f"Processing mode: {self.processing_mode}")
            self.logger.info(f"Total images: {image_analysis.get('total_images', 0)}")
            self.logger.info(f"Total labels: {image_analysis.get('total_labels', 0)}")
            self.logger.info(f"Image formats: {image_analysis.get('image_formats', {})}")
            self.logger.info(f"Label format: {image_analysis.get('label_format', 'unknown')}")
            self.logger.info(f"Classes: {len(image_analysis.get('classes', []))}")

            return {
                "processing_mode": self.processing_mode,
                "images_path": self.images_dir,
                "labels_path": self.labels_dir,
                "image_analysis": image_analysis
            }

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

    def identify_column_types(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Identify different column types including image columns

        Returns:
            Tuple of (numerical_cols, categorical_cols, textual_cols, image_cols)
        """
        section("IDENTIFYING COLUMN TYPES", self.logger)

        # Handle image mode
        if self.processing_mode == 'image':
            self.logger.info("Processing mode is image - using image data structure")
            image_cols = ['images', 'labels']  # Conceptual image columns
            self.feature_store_data['image_cols'] = image_cols
            return [], [], [], image_cols

        if self.df is None:
            self.logger.error("No data loaded. Please load data first.")
            return [], [], [], []

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
        self.feature_store_data['image_cols'] = []  # No image columns for tabular/textual data
        self.feature_store_data['id_cols'] = id_columns

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

        return numerical_cols, categorical_cols, textual_cols, []

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

        # Skip distribution analysis for image mode
        if self.processing_mode == 'image':
            self.logger.info("Skipping distribution analysis for image mode")
            return [], []

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

        # Skip outlier detection for image mode
        if self.processing_mode == 'image':
            self.logger.info("Skipping outlier detection for image mode")
            return []

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

        # Skip correlation analysis for image mode
        if self.processing_mode == 'image':
            self.logger.info("Skipping correlation analysis for image mode")
            return {}

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
                            'correlation': correlation_value
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
            target_col: Name of the target column or target type for images

        Returns:
            Name of the selected target column or None if invalid
        """
        section("TARGET COLUMN SELECTION", self.logger)

        if self.processing_mode == 'image':
            # For image mode, target_col might be a description like "classification" or "detection"
            self.feature_store_data['target_col'] = target_col
            self.logger.info(f"Target type set to: {target_col}")
            return target_col

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

        if not self.feature_store_data['target_col'] and self.processing_mode != 'image':
            self.logger.warning("Target column not set. Feature store will be incomplete.")

        try:
            # Convert any Python-specific types to standard types
            def convert_to_standard_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_standard_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_standard_types(v) for v in obj]
                elif isinstance(obj, tuple):
                    # Convert tuples to lists for better YAML compatibility
                    return [convert_to_standard_types(v) for v in obj]
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return obj

            # Convert the feature store data to standard types
            feature_store_data_converted = convert_to_standard_types(self.feature_store_data)

            yaml_path = os.path.join(self.feature_store_dir, 'feature_store.yaml')

            # Use safe_dump to avoid Python-specific tags and ensure clean output
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(
                    feature_store_data_converted,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    indent=2
                )

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
                'processing_mode': self.processing_mode,
                'processed_timestamp': datetime.now().strftime(DATETIME_FORMAT),
                'feature_store_path': os.path.join(self.feature_store_dir, 'feature_store.yaml'),
                'plots_dir': self.plots_dir,
                'target_column': self.feature_store_data['target_col']
            }

            # Add different paths based on processing mode
            if self.processing_mode == 'image':
                intel_data.update({
                    'original_file_name': self.feature_store_data.get('original_file_name', self.dataset_name),
                    'original_images_path': self.feature_store_data.get('original_images_path'),
                    'original_labels_path': self.feature_store_data.get('original_labels_path'),
                    'train_images_path': self.feature_store_data.get('train_images_path'),
                    'test_images_path': self.feature_store_data.get('test_images_path'),
                    'train_labels_path': self.feature_store_data.get('train_labels_path'),
                    'test_labels_path': self.feature_store_data.get('test_labels_path'),
                    'image_column': self.feature_store_data.get('image_column'),
                    'target_column': self.feature_store_data.get('target_column')
                })
            else:
                intel_data.update({
                    'original_file_name': self.feature_store_data.get('original_file_name', self.dataset_name),
                    'train_path': os.path.join(self.raw_data_dir, 'train.csv'),
                    'test_path': os.path.join(self.raw_data_dir, 'test.csv')
                })

            yaml_path = os.path.join(self.project_dir, 'intel.yaml')

            # Use safe_dump to avoid Python-specific tags
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(
                    intel_data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    indent=2
                )

            self.logger.info(f"Intel metadata saved to: {yaml_path}")
            return yaml_path

        except Exception as e:
            self.logger.error(f"Failed to save intel YAML: {e}")
            return None


    def generate_data_profile(self) -> Dict[str, str]:
        """
        Generate and save basic data profile plots

        Returns:
            Dictionary with plot paths
        """
        section("GENERATING DATA PROFILE", self.logger)

        if self.processing_mode == 'image':
            return self._generate_image_visualizations()

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
                for col in plot_columns:
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

                if len(numerical_cols) > 1:
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


    def _generate_image_visualizations(self) -> Dict[str, str]:
        """
        Generate visualizations specific to image data

        Returns:
            Dictionary with plot paths
        """
        self.logger.info("Generating image-specific visualizations")
        plot_paths = {}

        try:
            image_analysis = self.feature_store_data.get('image_analysis', {})

            # Create class distribution plot
            if 'class_distribution' in image_analysis and image_analysis['class_distribution']:
                plt.figure(figsize=(10, 6))
                classes = list(image_analysis['class_distribution'].keys())
                counts = list(image_analysis['class_distribution'].values())

                plt.bar(classes, counts)
                plt.title('Class Distribution')
                plt.xlabel('Classes')
                plt.ylabel('Number of Images')
                plt.xticks(rotation=45)

                # Save plot
                plot_path = os.path.join(self.plots_dir, 'class_distribution.png')
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()

                plot_paths['class_distribution'] = plot_path
                self.logger.info(f"Saved class distribution plot to {plot_path}")

            # Create image format distribution plot
            if 'image_formats' in image_analysis and image_analysis['image_formats']:
                plt.figure(figsize=(8, 6))
                formats = list(image_analysis['image_formats'].keys())
                counts = list(image_analysis['image_formats'].values())

                plt.pie(counts, labels=formats, autopct='%1.1f%%')
                plt.title('Image Format Distribution')

                # Save plot
                plot_path = os.path.join(self.plots_dir, 'image_formats.png')
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()

                plot_paths['image_formats'] = plot_path
                self.logger.info(f"Saved image formats plot to {plot_path}")

            # Create image size distribution plot
            if 'image_sizes' in image_analysis and image_analysis['image_sizes']:
                plt.figure(figsize=(12, 4))

                sizes = image_analysis['image_sizes']
                widths = [size[0] for size in sizes]
                heights = [size[1] for size in sizes]

                # Width distribution
                plt.subplot(1, 2, 1)
                plt.hist(widths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title('Image Width Distribution')
                plt.xlabel('Width (pixels)')
                plt.ylabel('Frequency')

                # Height distribution
                plt.subplot(1, 2, 2)
                plt.hist(heights, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                plt.title('Image Height Distribution')
                plt.xlabel('Height (pixels)')
                plt.ylabel('Frequency')

                # Save plot
                plot_path = os.path.join(self.plots_dir, 'image_dimensions.png')
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()

                plot_paths['image_dimensions'] = plot_path
                self.logger.info(f"Saved image dimensions plot to {plot_path}")

        except Exception as e:
            self.logger.warning(f"Error generating image visualizations: {e}")

        return plot_paths


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


    def perform_train_test_split(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Split data into training and testing sets

        Returns:
            Tuple of (train_df, test_df) or (None, None) for image mode
        """
        section("PERFORMING TRAIN-TEST SPLIT", self.logger)

        if self.processing_mode == 'image':
            return self._perform_image_train_test_split()

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


    def _perform_image_train_test_split(self) -> Tuple[None, None]:
        """
        Split image dataset into training and testing sets based on labels.csv

        Returns:
            Tuple of (None, None) as images are handled differently
        """
        section("PERFORMING IMAGE TRAIN-TEST SPLIT", self.logger)

        try:
            # Load the original labels CSV
            original_labels_path = self.feature_store_data['original_labels_path']
            self.logger.info(f"Loading original labels CSV from: {original_labels_path}")

            if not os.path.exists(original_labels_path):
                raise FileNotFoundError(f"Original labels CSV not found: {original_labels_path}")

            df = pd.read_csv(original_labels_path)
            self.logger.info(f"Loaded labels CSV with shape: {df.shape}")
            self.logger.info(f"Labels CSV columns: {list(df.columns)}")
            self.logger.info(f"Labels CSV head:\n{df.head()}")

            image_column = self.feature_store_data['image_column']
            target_column = self.feature_store_data['target_column']

            self.logger.info(f"Using image column: {image_column}")
            self.logger.info(f"Using target column: {target_column}")

            if not target_column:
                self.logger.warning("No target column detected, using random split")
                train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
                self.logger.info("Performed random train-test split")
            else:
                # Check target column values
                target_values = df[target_column].value_counts()
                self.logger.info(f"Target column value counts:\n{target_values}")

                # Stratified split based on target column
                self.logger.info(f"Attempting stratified split on target column: {target_column}")
                try:
                    train_df, test_df = train_test_split(
                        df,
                        test_size=TEST_SIZE,
                        random_state=RANDOM_STATE,
                        stratify=df[target_column]
                    )
                    self.logger.info("Successfully performed stratified train-test split")
                except ValueError as e:
                    # If stratification fails, use random split
                    self.logger.warning(f"Stratification failed ({e}), using random split")
                    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
                    self.logger.info("Performed random train-test split as fallback")

            self.logger.info(f"Train split shape: {train_df.shape}")
            self.logger.info(f"Test split shape: {test_df.shape}")

            # Log class distribution in splits
            if target_column:
                train_dist = train_df[target_column].value_counts()
                test_dist = test_df[target_column].value_counts()
                self.logger.info(f"Train class distribution:\n{train_dist}")
                self.logger.info(f"Test class distribution:\n{test_dist}")

            # Save train and test labels CSV
            train_labels_path = self.feature_store_data['train_labels_path']
            test_labels_path = self.feature_store_data['test_labels_path']

            self.logger.info(f"Saving train labels to: {train_labels_path}")
            self.logger.info(f"Saving test labels to: {test_labels_path}")

            train_df.to_csv(train_labels_path, index=False)
            test_df.to_csv(test_labels_path, index=False)

            # Verify the saved files
            train_verify = pd.read_csv(train_labels_path)
            test_verify = pd.read_csv(test_labels_path)
            self.logger.info(f"Verified train CSV: shape={train_verify.shape}")
            self.logger.info(f"Verified test CSV: shape={test_verify.shape}")

            # Copy corresponding images to train and test directories
            self.logger.info("Starting image copying process...")
            self._copy_images_by_labels(train_df, 'train')
            self._copy_images_by_labels(test_df, 'test')

            # Update feature store data
            self.feature_store_data['train_rows'] = len(train_df)
            self.feature_store_data['test_rows'] = len(test_df)

            self.logger.info("=== TRAIN-TEST SPLIT SUMMARY ===")
            self.logger.info(f"Total samples: {len(df)}")
            self.logger.info(f"Train samples: {len(train_df)}")
            self.logger.info(f"Test samples: {len(test_df)}")
            self.logger.info(f"Split ratio: {len(train_df) / len(df):.2f} / {len(test_df) / len(df):.2f}")
            self.logger.info("=================================")

            return None, None

        except Exception as e:
            self.logger.error(f"Failed to perform image train-test split: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None, None


    def _copy_images_by_labels(self, labels_df: pd.DataFrame, split_type: str):
        """
        Copy images from original directory to train/test directories based on labels DataFrame

        Args:
            labels_df: DataFrame containing image filenames and labels
            split_type: 'train' or 'test'
        """
        section(f"COPYING IMAGES FOR {split_type.upper()} SPLIT", self.logger)

        try:
            original_images_path = Path(self.feature_store_data['original_images_path'])
            image_column = self.feature_store_data['image_column']

            if split_type == 'train':
                target_images_dir = Path(self.feature_store_data['train_images_path'])
            else:
                target_images_dir = Path(self.feature_store_data['test_images_path'])

            self.logger.info(f"Source directory: {original_images_path}")
            self.logger.info(f"Target directory: {target_images_dir}")
            self.logger.info(f"Image column: {image_column}")
            self.logger.info(f"Processing {len(labels_df)} images for {split_type} split")

            # Ensure target directory exists
            target_images_dir.mkdir(parents=True, exist_ok=True)

            copied_count = 0
            missing_count = 0
            error_count = 0

            # Get list of all available images in source directory
            available_images = {}
            self.logger.info("Scanning source directory for available images...")
            for img_file in original_images_path.rglob('*'):
                if img_file.is_file() and self._is_image_file(str(img_file)):
                    # Store both original name and name without path
                    available_images[img_file.name.lower()] = img_file
                    # Also store with full relative path if it has subdirectories
                    rel_path = img_file.relative_to(original_images_path)
                    available_images[str(rel_path).lower()] = img_file

            self.logger.info(f"Found {len(set(available_images.values()))} unique images in source directory")
            self.logger.info(f"Sample available images: {list(available_images.keys())[:5]}")

            for idx, row in labels_df.iterrows():
                try:
                    image_filename = str(row[image_column]).strip()
                    self.logger.debug(f"Processing image {idx + 1}/{len(labels_df)}: {image_filename}")

                    # Handle different path formats
                    original_filename = image_filename
                    if '/' in image_filename or '\\' in image_filename:
                        # If it's a path, try both full path and just filename
                        image_basename = os.path.basename(image_filename)
                        search_names = [image_filename.lower(), image_basename.lower()]
                    else:
                        search_names = [image_filename.lower()]

                    # Find the image in the available images
                    source_image_path = None
                    matched_name = None

                    for search_name in search_names:
                        if search_name in available_images:
                            source_image_path = available_images[search_name]
                            matched_name = search_name
                            break

                        # Try without extension and with different extensions
                        name_without_ext = os.path.splitext(search_name)[0]
                        for ext in SUPPORTED_IMAGE_FORMATS:
                            potential_name = f"{name_without_ext}{ext}"
                            if potential_name in available_images:
                                source_image_path = available_images[potential_name]
                                matched_name = potential_name
                                break

                        if source_image_path:
                            break

                    if source_image_path and source_image_path.exists():
                        # Copy image to target directory, preserving original filename structure
                        target_filename = os.path.basename(original_filename)
                        target_image_path = target_images_dir / target_filename

                        # Ensure we don't overwrite existing files with different names
                        counter = 1
                        original_target = target_image_path
                        while target_image_path.exists() and target_image_path != source_image_path:
                            name_part = original_target.stem
                            ext_part = original_target.suffix
                            target_image_path = target_images_dir / f"{name_part}_{counter}{ext_part}"
                            counter += 1

                        shutil.copy2(source_image_path, target_image_path)
                        copied_count += 1

                        if idx < 5:  # Log first few successful copies
                            # Use regular check mark instead of Unicode to avoid encoding issues
                            self.logger.info(
                                f"[SUCCESS] Copied: {original_filename} -> {target_image_path.name} (matched: {matched_name})")
                    else:
                        missing_count += 1
                        if missing_count <= 10:  # Log first 10 missing files
                            self.logger.warning(f"[MISSING] Image not found: {image_filename} (searched: {search_names})")

                except Exception as e:
                    error_count += 1
                    self.logger.error(f"Error processing image {image_filename}: {e}")
                    if error_count <= 5:  # Log first 5 errors
                        self.logger.error(f"Error traceback: {traceback.format_exc()}")

            # Final summary
            self.logger.info(f"=== {split_type.upper()} IMAGE COPY SUMMARY ===")
            self.logger.info(f"Successfully copied: {copied_count} images")
            self.logger.info(f"Missing images: {missing_count}")
            self.logger.info(f"Errors: {error_count}")
            self.logger.info(f"Success rate: {copied_count / len(labels_df) * 100:.1f}%")

            if missing_count > 0:
                self.logger.warning(f"{missing_count} images were not found and skipped")
            if error_count > 0:
                self.logger.warning(f"{error_count} images had processing errors")

            # Verify final count
            actual_copied = len([f for f in target_images_dir.iterdir()
                                 if f.is_file() and self._is_image_file(str(f))])
            self.logger.info(f"Verification: {actual_copied} images actually exist in target directory")
            self.logger.info("=====================================")

        except Exception as e:
            self.logger.error(f"Error copying images for {split_type}: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise


    def run_ingestion_pipeline(self, file: Union[BinaryIO, str] = None, filename: str = None,
                               target_col: str = None, mode: str = None,
                               images_zip: Union[BinaryIO, str] = None,
                               labels_csv: Union[BinaryIO, str] = None) -> Dict:
        """
        Run the complete data ingestion pipeline

        Args:
            file: File-like object or path to the CSV file (for tabular/textual data)
            filename: Original filename (if file is file-like object)
            target_col: Name of the target column or target type
            mode: Processing mode ('tabular', 'textual', 'image', 'mixed', or None for auto-detect)
            images_zip: Zip file containing images (for image mode)
            labels_csv: CSV file containing labels (for image mode)

        Returns:
            Dictionary containing feature store information
        """
        section("STARTING DATA INGESTION PIPELINE", self.logger, char='*', length=80)

        try:
            # Handle different modes
            if mode == 'image' or (images_zip and labels_csv):
                # Image processing pipeline
                if not (images_zip and labels_csv):
                    raise ValueError("Both images_zip and labels_csv are required for image mode")

                # Load image dataset
                result = self.ingest_image_dataset(images_zip, labels_csv, filename)

                # Set target if provided (will use detected target from CSV if not provided)
                if target_col:
                    self.set_target_column(target_col)
                else:
                    # Use detected target column
                    detected_target = self.feature_store_data.get('target_column')
                    if detected_target:
                        self.set_target_column(detected_target)

            else:
                # Tabular/textual processing pipeline
                if not file:
                    raise ValueError("File is required for tabular/textual modes")

                # Step 1: Load the file with mode detection/setting
                self.ingest_uploaded_file(file, filename, mode)

                # Step 2: Display basic information
                data_info = self.display_data_info()

                # Step 3: Identify column types (including ID and textual columns)
                numerical_cols, categorical_cols, textual_cols, image_cols = self.identify_column_types()

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

            # Common steps for all modes
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
            if self.processing_mode == 'image':
                image_analysis = self.feature_store_data.get('image_analysis', {})
                results = {
                    "dataset_name": self.dataset_name,
                    "processing_mode": self.processing_mode,
                    "original_images_path": self.feature_store_data.get('original_images_path'),
                    "original_labels_path": self.feature_store_data.get('original_labels_path'),
                    "train_images_path": self.feature_store_data.get('train_images_path'),
                    "test_images_path": self.feature_store_data.get('test_images_path'),
                    "train_labels_path": self.feature_store_data.get('train_labels_path'),
                    "test_labels_path": self.feature_store_data.get('test_labels_path'),
                    "total_images": image_analysis.get('total_images', 0),
                    "total_labels": image_analysis.get('total_labels', 0),
                    "classes": image_analysis.get('classes', []),
                    "class_distribution": image_analysis.get('class_distribution', {}),
                    "image_formats": image_analysis.get('image_formats', {}),
                    "image_column": image_analysis.get('image_column'),
                    "target_column": image_analysis.get('target_column'),
                    "feature_store_path": feature_store_path,
                    "intel_path": intel_path,
                    "plots_dir": self.plots_dir,
                    "feature_store_data": self.feature_store_data
                }
            else:
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