#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering script for classification automl clone.
This module handles automatic feature generation, transformation pipeline creation,
and integration with preprocessing pipeline including handling imbalanced datasets.
API-friendly version that can be called from FastAPI endpoints.
"""
import matplotlib
from sklearn.linear_model import LogisticRegression

matplotlib.use('Agg')
import os
import sys
import yaml
import numpy as np
import pandas as pd
import cloudpickle
from typing import Dict, List, Union, Tuple, Optional
from pathlib import Path
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.validation import check_is_fitted

# Add parent directory to path for importing custom logger
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import custom logger
import logging
from semiauto_classification.logger import section, configure_logger
from semiauto_classification.custom_transformers import FeatureToolsTransformer, IdentityTransformer, SHAPFeatureSelector


class FeatureEngineer:
    """Main class for feature engineering process, API-friendly version."""

    def __init__(self, config_path: Union[str, Path] = "intel.yaml"):
        """
        Initialize the FeatureEngineer.

        Args:
            config_path: Path to the config file (intel.yaml)
        """
        self.logger = logging.getLogger("Feature Engineering")
        section("FEATURE ENGINEERING INITIALIZATION", self.logger)

        # Configure logger if not already configured
        try:
            if not self.logger.handlers:
                configure_logger()
        except Exception:
            # If logger configuration fails, continue with default logger
            pass

        self.config_path = Path(config_path)
        self.project_root = self.config_path.parent
        self.intel = self._load_intel()

        # Load dataset name from config
        self.dataset_name = self.intel.get("dataset_name")
        if not self.dataset_name:
            raise ValueError("dataset_name not found in intel.yaml")

        self.feature_store = self._load_feature_store()
        self.target_column = self.intel.get("target_column")
        self._setup_paths()

    def _load_intel(self) -> Dict:
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading intel.yaml: {str(e)}")
            raise

    def _load_feature_store(self) -> Dict:
        try:
            feature_store_path = self.project_root / self.intel.get("feature_store_path")
            with open(feature_store_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading feature store: {str(e)}")
            raise

    def _setup_paths(self):
        # Construct absolute paths using project root
        self.transformation_pipeline_path = self.project_root / f"model/pipelines/preprocessing_{self.dataset_name}/transformation.pkl"
        self.processor_pipeline_path = self.project_root / f"model/pipelines/preprocessing_{self.dataset_name}/processor.pkl"
        self.train_transformed_path = self.project_root / f"data/processed/data_{self.dataset_name}/train_transformed.csv"
        self.test_transformed_path = self.project_root / f"data/processed/data_{self.dataset_name}/test_transformed.csv"

        # Ensure directories exist
        self.transformation_pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        self.train_transformed_path.parent.mkdir(parents=True, exist_ok=True)

    def _update_intel(self, resampling_method: Optional[str], use_feature_tools: bool, use_shap: bool, n_features: int):
        self.intel.update({
            "transformation_pipeline_path": str(self.transformation_pipeline_path.relative_to(self.project_root)),
            "processor_pipeline_path": str(self.processor_pipeline_path.relative_to(self.project_root)),
            "train_transformed_path": str(self.train_transformed_path.relative_to(self.project_root)),
            "test_transformed_path": str(self.test_transformed_path.relative_to(self.project_root)),
            "feature_engineering_config": {
                "resampling_method": resampling_method,
                "use_feature_tools": use_feature_tools,
                "use_shap_selection": use_shap,
                "n_features_selected": n_features if use_shap else None,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        })
        with open(self.config_path, 'w') as f:
            yaml.dump(self.intel, f)
        self.logger.info(f"Updated intel.yaml at {self.config_path}")
        return self.intel

    def run(self, resampling_method: Optional[str] = None, use_feature_tools: bool = False, use_shap: bool = False, n_features: int = 20):
        """
        Run the feature engineering process with the specified parameters.
        """
        section("FEATURE ENGINEERING PROCESS", self.logger)
        train_df, test_df = self._load_data()
        X_train, y_train = train_df.drop(columns=[self.target_column]), train_df[self.target_column]
        X_test, y_test = test_df.drop(columns=[self.target_column]), test_df[self.target_column]

        pipeline_steps = []
        # Add resampling step if specified
        if resampling_method:
            if resampling_method == 'SMOTE':
                resampler = SMOTE(random_state=42)
            elif resampling_method == 'SMOTETomek':
                resampler = SMOTETomek(random_state=42)
            elif resampling_method == 'SMOTEENN':
                resampler = SMOTEENN(random_state=42)
            elif resampling_method == 'ADASYN':
                resampler = ADASYN(random_state=42)
            else:
                raise ValueError(f"Invalid resampling method: {resampling_method}")
            pipeline_steps.append(('resampler', resampler))

        if use_feature_tools:
            pipeline_steps.append(('feature_tools', FeatureToolsTransformer(self.target_column)))
        else:
            pipeline_steps.append(('identity', IdentityTransformer()))

        if use_shap:
            pipeline_steps.append(('shap_selector', SHAPFeatureSelector(n_features=n_features)))

        transformation_pipeline = Pipeline(pipeline_steps)
        transformation_pipeline.fit(X_train, y_train)

        try:
            X_train_transformed = transformation_pipeline.transform(X_train).reset_index(drop=True)
            X_test_transformed = transformation_pipeline.transform(X_test).reset_index(drop=True)
            y_train_reset = y_train.reset_index(drop=True)
            y_test_reset = y_test.reset_index(drop=True)

            train_transformed_df = pd.concat([X_train_transformed, y_train_reset], axis=1)
            test_transformed_df = pd.concat([X_test_transformed, y_test_reset], axis=1)

            self._save_data(train_transformed_df, test_transformed_df)
            self._save_pipelines(transformation_pipeline)
            self._log_feature_info(transformation_pipeline, use_feature_tools, use_shap)
            updated_intel = self._update_intel(resampling_method, use_feature_tools, use_shap, n_features)

            return {
                "status": "success",
                "message": "Feature engineering completed successfully",
                "metadata": {
                    "train_shape": train_transformed_df.shape,
                    "test_shape": test_transformed_df.shape,
                    "train_path": str(self.train_transformed_path),
                    "test_path": str(self.test_transformed_path),
                    "pipeline_path": str(self.transformation_pipeline_path),
                    "processor_path": str(self.processor_pipeline_path),
                    "feature_engineering_config": updated_intel.get("feature_engineering_config", {})
                }
            }

        except Exception as e:
            error_msg = f"Error in transformation process: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }

    def _load_data(self):
        try:
            train_path = self.project_root / self.intel.get("train_preprocessed_path")
            test_path = self.project_root / self.intel.get("test_preprocessed_path")
            self.logger.info(f"Loading train data from {train_path}")
            self.logger.info(f"Loading test data from {test_path}")
            return (
                pd.read_csv(train_path),
                pd.read_csv(test_path)
            )
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _save_data(self, train_df, test_df):
        try:
            self.logger.info(f"Saving transformed train data to {self.train_transformed_path}")
            train_df.to_csv(self.train_transformed_path, index=False)
            self.logger.info(f"Saving transformed test data to {self.test_transformed_path}")
            test_df.to_csv(self.test_transformed_path, index=False)
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise

    def _load_cleaning_pipeline(self):
        try:
            cleaning_path = self.project_root / f"model/pipelines/preprocessing_{self.dataset_name}/cleaning.pkl"
            self.logger.info(f"Loading cleaning pipeline from {cleaning_path}")
            with open(cleaning_path, 'rb') as f:
                return cloudpickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading cleaning pipeline: {str(e)}")
            raise

    def _load_preprocessing_pipeline(self):
        try:
            preprocessing_path = self.project_root / self.intel.get("preprocessing_pipeline_path")
            self.logger.info(f"Loading preprocessing pipeline from {preprocessing_path}")
            with open(preprocessing_path, 'rb') as f:
                return cloudpickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading preprocessing pipeline: {str(e)}")
            raise

    # Replace the _save_pipelines method in feature_engineering.py
    def _save_pipelines(self, transformation_pipeline):
        try:
            self.logger.info(f"Saving transformation pipeline to {self.transformation_pipeline_path}")

            # Save with custom reducer
            with open(self.transformation_pipeline_path, 'wb') as f:
                cloudpickle.dump(transformation_pipeline, f)

            # Load cleaning and preprocessing pipelines
            cleaning_pipeline = self._load_cleaning_pipeline()
            preprocessing_pipeline = self._load_preprocessing_pipeline()

            # Create a combined pipeline with all three components
            processor_pipeline = Pipeline([
                ('cleaning', cleaning_pipeline),
                ('preprocessing', preprocessing_pipeline),
                ('transformation', transformation_pipeline)
            ])

            self.logger.info(f"Saving processor pipeline to {self.processor_pipeline_path}")

            # Save with custom reducer
            with open(self.processor_pipeline_path, 'wb') as f:
                cloudpickle.dump(processor_pipeline, f)

        except Exception as e:
            self.logger.error(f"Error saving pipelines: {str(e)}")
            raise

    def _log_feature_info(self, pipeline, use_feature_tools, use_shap):
        if use_feature_tools and 'feature_tools' in pipeline.named_steps:
            self.logger.info(f"Generated {len(pipeline.named_steps['feature_tools'].feature_names)} features")

        if use_shap and 'shap_selector' in pipeline.named_steps:
            selector = pipeline.named_steps['shap_selector']
            if selector.importance_df is not None:
                top_features = selector.importance_df.head(10)['feature'].tolist()
                self.logger.info(f"Top 10 features: {top_features}")

    def get_feature_importance(self):
        """
        Return feature importance if SHAP selection was used.

        Returns:
            Dict: Feature importance data or error message
        """
        try:
            # Try to load the transformation pipeline
            with open(self.transformation_pipeline_path, 'rb') as f:
                pipeline = cloudpickle.load(f)

            if 'shap_selector' in pipeline.named_steps:
                selector = pipeline.named_steps['shap_selector']
                if selector.importance_df is not None:
                    return {
                        "status": "success",
                        "feature_importance": selector.importance_df.to_dict(orient='records')
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Feature importance data not available"
                    }
            else:
                return {
                    "status": "error",
                    "message": "SHAP feature selection was not used"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error retrieving feature importance: {str(e)}"
            }

    def get_generated_features(self):
        """
        Return generated features if FeatureTools was used.

        Returns:
            Dict: Generated features data or error message
        """
        try:
            # Try to load the transformation pipeline
            with open(self.transformation_pipeline_path, 'rb') as f:
                pipeline = cloudpickle.load(f)

            if 'feature_tools' in pipeline.named_steps:
                feature_tools = pipeline.named_steps['feature_tools']
                if feature_tools.feature_names is not None:
                    return {
                        "status": "success",
                        "generated_features": feature_tools.feature_names
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Generated features not available"
                    }
            else:
                return {
                    "status": "error",
                    "message": "FeatureTools was not used"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error retrieving generated features: {str(e)}"
            }


# Helper function for API usage
def run_feature_engineering(
        config_path: str = "intel.yaml",
        resampling_method: Optional[str] = None,
        use_feature_tools: bool = False,
        use_shap: bool = False,
        n_features: int = 20
):
    """
    Run feature engineering with the specified parameters.
    This function can be called from an API endpoint.

    Args:
        config_path: Path to the config file (intel.yaml)
        resampling_method: Optional resampling method (SMOTE, SMOTETomek, SMOTEENN, ADASYN)
        use_feature_tools: Whether to use FeatureTools for feature generation
        use_shap: Whether to use SHAP for feature selection
        n_features: Number of features to select if using SHAP

    Returns:
        Dict: Result of the feature engineering process
    """
    try:
        engineer = FeatureEngineer(config_path=config_path)
        result = engineer.run(
            resampling_method=resampling_method,
            use_feature_tools=use_feature_tools,
            use_shap=use_shap,
            n_features=n_features
        )
        return result
    except Exception as e:
        logger = logging.getLogger("Feature Engineering")
        logger.error(f"Error in feature engineering: {str(e)}")
        return {
            "status": "error",
            "message": f"Feature engineering failed: {str(e)}"
        }


if __name__ == "__main__":
    # This block is for direct script execution (not API call)
    # It demonstrates how to use the API-friendly version
    try:
        # Get parameters from command line arguments if provided
        import argparse

        parser = argparse.ArgumentParser(description='Run feature engineering process')
        parser.add_argument('--config', default='intel.yaml', help='Path to config file')
        parser.add_argument('--resampling-method', choices=['SMOTE', 'SMOTETomek', 'SMOTEENN', 'ADASYN'],
                          help='Resampling method for imbalanced data')
        parser.add_argument('--use-feature-tools', action='store_true', help='Use FeatureTools')
        parser.add_argument('--use-shap', action='store_true', help='Use SHAP feature selection')
        parser.add_argument('--n-features', type=int, default=20, help='Number of features to select')
        args = parser.parse_args()

        # Configure logger
        configure_logger()
        logger = logging.getLogger("Feature Engineering")

        # Run feature engineering
        result = run_feature_engineering(
            config_path=args.config,
            resampling_method=args.resampling_method,
            use_feature_tools=args.use_feature_tools,
            use_shap=args.use_shap,
            n_features=args.n_features
        )

        if result["status"] == "success":
            logger.info("Feature engineering completed successfully")
        else:
            logger.critical(f"Feature engineering failed: {result['message']}")
            sys.exit(1)

    except Exception as e:
        logger = logging.getLogger("Feature Engineering")
        logger.critical(f"Feature engineering failed: {str(e)}")
        sys.exit(1)