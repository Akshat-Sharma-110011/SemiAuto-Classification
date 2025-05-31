"""
Model Evaluation Module for Classification.

This module evaluates the performance of trained classification model using various metrics
and stores the results in a YAML file. It also evaluates the optimized model if available.
API-friendly version that can be imported and used in a FastAPI application.
"""

import os
import yaml
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# Classification metrics imports
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    log_loss,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score
)

# Import custom logger
import logging
from semiauto_classification.logger import section, configure_logger  # Configure logger

# Configure logger
configure_logger()
logger = logging.getLogger("Model Evaluation")


def load_intel(intel_path: str = "intel.yaml") -> Dict[str, Any]:
    """
    Load the intelligence YAML file containing paths and configurations.

    Args:
        intel_path: Path to the intel YAML file

    Returns:
        Dictionary containing the loaded intel data
    """
    section(f"Loading Intel from {intel_path}", logger)
    try:
        with open(intel_path, "r") as f:
            intel = yaml.safe_load(f)
        logger.info(f"Successfully loaded intel from {intel_path}")
        return intel
    except Exception as e:
        logger.error(f"Failed to load intel file: {e}")
        raise


def load_model(model_path: str) -> Any:
    """
    Load a trained model from the specified path.

    Args:
        model_path: Path to the saved model file

    Returns:
        Loaded model object
    """
    section(f"Loading Model from {model_path}", logger)
    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_test_data(test_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the test dataset and separate features from target.

    Args:
        test_path: Path to the test dataset
        target_column: Name of the target column

    Returns:
        Tuple of (X_test, y_test)
    """
    section(f"Loading Test Data from {test_path}", logger)
    try:
        test_data = pd.read_csv(test_path)
        logger.info(f"Test data shape: {test_data.shape}")

        # Split features and target
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]

        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        logger.info(f"Number of unique classes: {y_test.nunique()}")
        logger.info(f"Class distribution: {y_test.value_counts().to_dict()}")

        return X_test, y_test
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise


def is_binary_classification(y_true: pd.Series) -> bool:
    """
    Check if the classification problem is binary or multiclass.

    Args:
        y_true: True target values

    Returns:
        True if binary classification, False if multiclass
    """
    return len(np.unique(y_true)) == 2


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate the classification model using various metrics.

    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test target values

    Returns:
        Dictionary containing metric names and values
    """
    section("Evaluating Classification Model Performance", logger)
    try:
        # Make predictions
        y_pred = model.predict(X_test)

        # Check if model has predict_proba method for probability-based metrics
        has_predict_proba = hasattr(model, 'predict_proba')
        if has_predict_proba:
            y_pred_proba = model.predict_proba(X_test)

        # Determine if binary or multiclass
        is_binary = is_binary_classification(y_test)

        # Calculate basic classification metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "matthews_corrcoef": float(matthews_corrcoef(y_test, y_pred)),
            "cohen_kappa": float(cohen_kappa_score(y_test, y_pred))
        }

        # Add precision, recall, and f1-score
        if is_binary:
            metrics.update({
                "precision": float(precision_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred)),
                "f1_score": float(f1_score(y_test, y_pred)),
                "specificity": float(recall_score(y_test, y_pred, pos_label=0))  # True Negative Rate
            })
        else:
            # For multiclass, use macro average
            metrics.update({
                "precision_macro": float(precision_score(y_test, y_pred, average='macro')),
                "recall_macro": float(recall_score(y_test, y_pred, average='macro')),
                "f1_score_macro": float(f1_score(y_test, y_pred, average='macro')),
                "precision_weighted": float(precision_score(y_test, y_pred, average='weighted')),
                "recall_weighted": float(recall_score(y_test, y_pred, average='weighted')),
                "f1_score_weighted": float(f1_score(y_test, y_pred, average='weighted'))
            })

        # Add probability-based metrics if available
        if has_predict_proba:
            try:
                # Log loss (cross-entropy)
                metrics["log_loss"] = float(log_loss(y_test, y_pred_proba))

                # AUC metrics
                if is_binary:
                    # Binary classification AUC
                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
                else:
                    # Multiclass AUC (one-vs-rest)
                    try:
                        metrics["roc_auc_ovr"] = float(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))
                        metrics["roc_auc_ovo"] = float(roc_auc_score(y_test, y_pred_proba, multi_class='ovo'))
                    except ValueError as e:
                        logger.warning(f"Could not calculate multiclass AUC: {e}")

            except Exception as e:
                logger.warning(f"Could not calculate probability-based metrics: {e}")

        # Add confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()  # Convert to list for YAML serialization

        # Add classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # Convert numpy values to float for YAML serialization
        def convert_numpy_values(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_values(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj

        metrics["classification_report"] = convert_numpy_values(class_report)

        # Add problem type and class information
        metrics["problem_type"] = "binary" if is_binary else "multiclass"
        metrics["num_classes"] = int(len(np.unique(y_test)))
        metrics["class_names"] = [str(cls) for cls in sorted(np.unique(y_test))]

        # Log key metrics
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")

        if is_binary:
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
            if 'roc_auc' in metrics:
                logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        else:
            logger.info(f"Precision (Macro): {metrics['precision_macro']:.4f}")
            logger.info(f"Recall (Macro): {metrics['recall_macro']:.4f}")
            logger.info(f"F1-Score (Macro): {metrics['f1_score_macro']:.4f}")

        logger.info(f"Matthews Correlation Coefficient: {metrics['matthews_corrcoef']:.4f}")
        logger.info(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")

        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


def save_metrics(metrics: Dict[str, Any], dataset_name: str, filename: str = "performance.yaml") -> str:
    """
    Save metrics to a YAML file in the reports/metrics directory.

    Args:
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset
        filename: Name of the metrics file (default: performance.yaml)

    Returns:
        Path to the saved metrics file
    """
    section(f"Saving Performance Metrics to {filename}", logger)
    try:
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.join("reports", "metrics", f"performance_{dataset_name}")
        os.makedirs(metrics_dir, exist_ok=True)

        # Add timestamp to metrics
        metrics["evaluation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Convert numpy values to native Python types to prevent YAML serialization issues
        cleaned_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                cleaned_metrics[key] = value.tolist()
            elif isinstance(value, np.number):
                cleaned_metrics[key] = float(value)
            else:
                cleaned_metrics[key] = value

        # Define metrics file path
        metrics_file_path = os.path.join(metrics_dir, filename)

        # Save metrics to YAML
        with open(metrics_file_path, "w") as f:
            yaml.dump(cleaned_metrics, f, default_flow_style=False, indent=2)

        logger.info(f"Metrics saved to {metrics_file_path}")
        return metrics_file_path
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        raise


def update_intel(intel: Dict[str, Any], metrics_path: str, intel_path: str = "intel.yaml",
                 is_optimized: bool = False) -> Dict[str, Any]:
    """
    Update the intel YAML file with the metrics file path.

    Args:
        intel: Dictionary containing intel data
        metrics_path: Path to the saved metrics file
        intel_path: Path to the intel YAML file
        is_optimized: Whether the metrics are for the optimized model

    Returns:
        Updated intel dictionary
    """
    section("Updating Intel YAML", logger)
    try:
        # Create a new intel dictionary to avoid modifying the original
        updated_intel = intel.copy()

        # Update intel dictionary with appropriate key based on model type
        if is_optimized:
            updated_intel["optimized_performance_metrics_path"] = metrics_path
            updated_intel["optimized_evaluation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Intel updated with optimized performance metrics path: {metrics_path}")
        else:
            updated_intel["performance_metrics_path"] = metrics_path
            updated_intel["evaluation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Intel updated with performance metrics path: {metrics_path}")

        # Save updated intel to YAML
        with open(intel_path, "w") as f:
            yaml.dump(updated_intel, f, default_flow_style=False)

        return updated_intel

    except Exception as e:
        logger.error(f"Failed to update intel: {e}")
        raise


def check_optimized_model_exists(dataset_name: str) -> Tuple[bool, str]:
    """
    Check if the optimized model exists.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Tuple of (exists, path)
    """
    optimized_model_path = os.path.join("model", f"model_{dataset_name}", "optimized_model.pkl")
    exists = os.path.isfile(optimized_model_path)

    if exists:
        logger.info(f"Optimized model found at {optimized_model_path}")
    else:
        logger.info("No optimized model found")

    return exists, optimized_model_path


def evaluate_and_save_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                            dataset_name: str, is_optimized: bool = False) -> Dict[str, Any]:
    """
    Evaluate a model and save its metrics.

    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test target values
        dataset_name: Name of the dataset
        is_optimized: Whether this is an optimized model

    Returns:
        Dictionary of evaluation metrics
    """
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Determine filename
    filename = "optimized_performance.yaml" if is_optimized else "performance.yaml"

    # Save metrics
    metrics_path = save_metrics(metrics, dataset_name, filename)

    # Return metrics with path
    metrics["metrics_path"] = metrics_path
    return metrics


def run_evaluation(intel_path: str = "intel.yaml") -> Dict[str, Any]:
    """
    Run the complete model evaluation process.

    Args:
        intel_path: Path to the intel YAML file

    Returns:
        Dictionary containing evaluation results
    """
    section("Starting Classification Model Evaluation", logger, char="*", length=60)
    results = {
        "success": False,
        "standard_model": None,
        "optimized_model": None,
        "intel": None,
        "error": None
    }

    try:
        # Load intel
        intel = load_intel(intel_path)
        results["intel"] = intel

        # Extract required paths and config
        model_path = intel["model_path"]
        test_path = intel["test_transformed_path"]
        target_column = intel["target_column"]
        dataset_name = intel["dataset_name"]

        # Load test data - do this once for both evaluations
        X_test, y_test = load_test_data(test_path, target_column)

        # --- Evaluate the original model ---
        # Load original model
        model = load_model(model_path)

        # Evaluate and save original model
        standard_metrics = evaluate_and_save_model(model, X_test, y_test, dataset_name, is_optimized=False)
        results["standard_model"] = standard_metrics

        # Update intel with original metrics path
        intel = update_intel(intel, standard_metrics["metrics_path"], intel_path)
        results["intel"] = intel

        # --- Check for and evaluate the optimized model if it exists ---
        optimized_exists, optimized_model_path = check_optimized_model_exists(dataset_name)

        if optimized_exists:
            section("Evaluating Optimized Classification Model", logger, char="-", length=50)

            # Load optimized model
            optimized_model = load_model(optimized_model_path)

            # Evaluate and save optimized model
            optimized_metrics = evaluate_and_save_model(
                optimized_model, X_test, y_test, dataset_name, is_optimized=True)
            results["optimized_model"] = optimized_metrics

            # Update intel with optimized metrics path
            intel = update_intel(intel, optimized_metrics["metrics_path"], intel_path, is_optimized=True)
            results["intel"] = intel

            logger.info("Optimized model evaluation completed successfully")

        section("Classification Model Evaluation Complete", logger, char="*", length=60)
        results["success"] = True
        return results

    except Exception as e:
        error_msg = f"Classification model evaluation failed: {str(e)}"
        logger.critical(error_msg)
        section("Classification Model Evaluation Failed", logger, level=logging.CRITICAL, char="*", length=60)
        results["error"] = error_msg
        return results


def get_evaluation_summary(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a summary of the evaluation results.

    Args:
        evaluation_results: Results from run_evaluation

    Returns:
        Dictionary with summary information
    """
    summary = {
        "success": evaluation_results["success"],
        "dataset_name": evaluation_results["intel"]["dataset_name"] if evaluation_results["intel"] else None,
        "standard_model": {},
        "optimized_model": {},
        "has_optimized_model": evaluation_results["optimized_model"] is not None
    }

    # Extract key metrics for standard model
    if evaluation_results["standard_model"]:
        metrics = evaluation_results["standard_model"]
        is_binary = metrics.get("problem_type") == "binary"

        summary["standard_model"] = {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "problem_type": metrics["problem_type"],
            "num_classes": metrics["num_classes"]
        }

        if is_binary:
            summary["standard_model"].update({
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics.get("roc_auc")
            })
        else:
            summary["standard_model"].update({
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_score_macro": metrics["f1_score_macro"]
            })

    # Extract key metrics for optimized model if available
    if evaluation_results["optimized_model"]:
        metrics = evaluation_results["optimized_model"]
        is_binary = metrics.get("problem_type") == "binary"

        summary["optimized_model"] = {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "problem_type": metrics["problem_type"],
            "num_classes": metrics["num_classes"]
        }

        if is_binary:
            summary["optimized_model"].update({
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics.get("roc_auc")
            })
        else:
            summary["optimized_model"].update({
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_score_macro": metrics["f1_score_macro"]
            })

        # Add improvement metrics if both model exist
        if evaluation_results["standard_model"]:
            std_metrics = evaluation_results["standard_model"]
            opt_metrics = evaluation_results["optimized_model"]

            # Calculate improvement percentages
            accuracy_improvement = (opt_metrics["accuracy"] - std_metrics["accuracy"]) / max(std_metrics["accuracy"],
                                                                                             1e-10) * 100
            balanced_accuracy_improvement = (opt_metrics["balanced_accuracy"] - std_metrics["balanced_accuracy"]) / max(
                std_metrics["balanced_accuracy"], 1e-10) * 100

            improvement = {
                "accuracy": accuracy_improvement,
                "balanced_accuracy": balanced_accuracy_improvement
            }

            if is_binary:
                f1_improvement = (opt_metrics["f1_score"] - std_metrics["f1_score"]) / max(std_metrics["f1_score"],
                                                                                           1e-10) * 100
                improvement["f1_score"] = f1_improvement

                if opt_metrics.get("roc_auc") and std_metrics.get("roc_auc"):
                    auc_improvement = (opt_metrics["roc_auc"] - std_metrics["roc_auc"]) / max(std_metrics["roc_auc"],
                                                                                              1e-10) * 100
                    improvement["roc_auc"] = auc_improvement
            else:
                f1_macro_improvement = (opt_metrics["f1_score_macro"] - std_metrics["f1_score_macro"]) / max(
                    std_metrics["f1_score_macro"], 1e-10) * 100
                improvement["f1_score_macro"] = f1_macro_improvement

            summary["improvement"] = improvement

    return summary


if __name__ == "__main__":
    # This block only runs when the script is executed directly
    results = run_evaluation()
    if results["success"]:
        print("Classification model evaluation completed successfully.")
        # Print summary of evaluation results
        summary = get_evaluation_summary(results)
        print(f"\nEvaluation Summary for {summary['dataset_name']}:")
        print(f"Problem Type: {summary['standard_model'].get('problem_type', 'Unknown')}")
        print(f"Number of Classes: {summary['standard_model'].get('num_classes', 'Unknown')}")
        print(f"Standard Model Accuracy: {summary['standard_model']['accuracy']:.4f}")
        print(f"Standard Model Balanced Accuracy: {summary['standard_model']['balanced_accuracy']:.4f}")

        if summary['standard_model'].get('problem_type') == 'binary':
            print(f"Standard Model F1-Score: {summary['standard_model']['f1_score']:.4f}")
            if summary['standard_model'].get('roc_auc'):
                print(f"Standard Model ROC AUC: {summary['standard_model']['roc_auc']:.4f}")
        else:
            print(f"Standard Model F1-Score (Macro): {summary['standard_model']['f1_score_macro']:.4f}")

        if summary['has_optimized_model']:
            print(f"\nOptimized Model Accuracy: {summary['optimized_model']['accuracy']:.4f}")
            print(f"Optimized Model Balanced Accuracy: {summary['optimized_model']['balanced_accuracy']:.4f}")

            if summary['optimized_model'].get('problem_type') == 'binary':
                print(f"Optimized Model F1-Score: {summary['optimized_model']['f1_score']:.4f}")
                if summary['optimized_model'].get('roc_auc'):
                    print(f"Optimized Model ROC AUC: {summary['optimized_model']['roc_auc']:.4f}")
            else:
                print(f"Optimized Model F1-Score (Macro): {summary['optimized_model']['f1_score_macro']:.4f}")

            if 'improvement' in summary:
                print(f"\nImprovements:")
                print(f"Accuracy: {summary['improvement']['accuracy']:.2f}%")
                print(f"Balanced Accuracy: {summary['improvement']['balanced_accuracy']:.2f}%")
                if 'f1_score' in summary['improvement']:
                    print(f"F1-Score: {summary['improvement']['f1_score']:.2f}%")
                if 'f1_score_macro' in summary['improvement']:
                    print(f"F1-Score (Macro): {summary['improvement']['f1_score_macro']:.2f}%")
                if 'roc_auc' in summary['improvement']:
                    print(f"ROC AUC: {summary['improvement']['roc_auc']:.2f}%")
    else:
        print(f"Classification model evaluation failed: {results['error']}")