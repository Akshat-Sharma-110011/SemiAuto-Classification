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
import sys

# Import the logger
from semiauto_classification.logger import section

# Suppress specific pandas warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Get module level logger
logger = logging.getLogger(__name__)
from semiauto_classification.custom_transformers import DataCleaner, CleaningParameters


def main(dataset_name: Optional[str] = None):
    try:
        logger.info("Starting data cleaning process")
        section("Data Cleaning Process", logger)

        # Initialize the data cleaner
        cleaner = DataCleaner()

        # Load configuration
        intel = cleaner._get_intel_config()
        if dataset_name is None:
            dataset_name = intel.get('dataset_name')
            if not dataset_name:
                # Try to infer dataset name from directory structure
                base_dir = Path(__file__).resolve().parent.parent.parent
                raw_data_dir = base_dir / 'data' / 'raw'
                if raw_data_dir.exists():
                    dataset_folders = [d.name for d in raw_data_dir.iterdir()
                                       if d.is_dir() and d.name.startswith('data_')]
                    if dataset_folders:
                        dataset_name = dataset_folders[0].replace('data_', '')
                        logger.warning(f"Inferred dataset name: {dataset_name}")

                if not dataset_name:
                    raise ValueError("Dataset name not provided and not found in intel.yaml")

        logger.info(f"Processing dataset: {dataset_name}")

        # Validate and create paths - UPDATED PATH CALCULATION
        base_dir = Path(__file__).resolve().parent.parent.parent  # Now points to project root
        raw_data_dir = base_dir / 'data' / 'raw' / f'data_{dataset_name}'
        cleaned_dir = base_dir / 'data' / 'cleaned' / f'data_{dataset_name}'
        pipeline_dir = base_dir / 'model' / 'pipelines' / f'preprocessing_{dataset_name}'
        reports_dir = base_dir / 'reports' / 'readme'

        # Create necessary directories
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Construct file paths
        train_path = raw_data_dir / 'train.csv'
        test_path = raw_data_dir / 'test.csv'

        # Update intel.yaml with correct paths
        cleaner._update_intel_config(dataset_name, {
            'train_path': str(train_path),
            'test_path': str(test_path)
        })

        # Check if files exist
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}. "
                                    "Please ensure the file exists or run data splitting first.")

        has_test = test_path.exists()
        if not has_test:
            logger.warning(f"Test data not found at {test_path}, will only process training data")

        # Load training data
        logger.info(f"Loading training data from {train_path}")
        try:
            train_df = pd.read_csv(train_path, low_memory=False)
            logger.info(f"Loaded training data: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
        except Exception as e:
            raise IOError(f"Failed to load training data: {str(e)}") from e

        # Load test data if available
        test_df = None
        if has_test:
            logger.info(f"Loading test data from {test_path}")
            try:
                test_df = pd.read_csv(test_path, low_memory=False)
                logger.info(f"Loaded test data: {test_df.shape[0]} rows × {test_df.shape[1]} columns")
            except Exception as e:
                logger.error(f"Failed to load test data: {str(e)}")
                has_test = False

        # Fit and transform data
        logger.info("Fitting cleaning pipeline on training data")
        cleaner.fit(train_df, dataset_name)

        logger.info("Transforming training data")
        cleaned_train = cleaner.transform(train_df)

        cleaned_test = None
        if has_test and test_df is not None:
            logger.info("Transforming test data")
            cleaned_test = cleaner.transform(test_df)

        # Save cleaned data
        cleaned_train_path = cleaned_dir / "train.csv"
        logger.info(f"Saving cleaned training data to {cleaned_train_path}")
        cleaned_train.to_csv(cleaned_train_path, index=False)

        if has_test and cleaned_test is not None:
            cleaned_test_path = cleaned_dir / "test.csv"
            logger.info(f"Saving cleaned test data to {cleaned_test_path}")
            cleaned_test.to_csv(cleaned_test_path, index=False)

        # Save pipeline
        cleaning_pipeline_path = pipeline_dir / "cleaning.pkl"
        logger.info(f"Saving cleaning pipeline to {cleaning_pipeline_path}")
        cleaner.save(str(cleaning_pipeline_path))

        # Generate report
        report_path = reports_dir / "cleaning_report.md"
        logger.info(f"Generating cleaning report at {report_path}")
        cleaner.save_cleaning_report(str(report_path))

        # Update intel.yaml
        update_dict = {
            'cleaned_train_path': str(cleaned_train_path),
            'cleaning_pipeline_path': str(cleaning_pipeline_path),
            'cleaning_report_path': str(report_path),
            'cleaning_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if has_test:
            update_dict['cleaned_test_path'] = str(cleaned_test_path)

        cleaner._update_intel_config(dataset_name, update_dict)

        logger.info("Data cleaning completed successfully")
        return True

    except Exception as e:
        logger.error(f"Critical error in data cleaning process: {str(e)}")
        logger.error("Data cleaning failed. Check error details above.")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean raw data for machine learning.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset to clean")

    args = parser.parse_args()
    success = main(args.dataset)
    sys.exit(0 if success else 1)
