import os
import yaml
import cloudpickle
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any, Tuple, List, Optional, Union

# Import optimization libraries
from sklearn.model_selection import GridSearchCV
import optuna
import atexit
from catboost.core import _custom_loggers_stack

# 1. UPDATE IMPORTS - Replace regression metrics with classification metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    log_loss
)

# 2. UPDATE MODEL IMPORTS - Replace regressors with classifiers
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, VotingClassifier, BaggingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Import custom logger
from semiauto_classification.logger import section, configure_logger

INTEL_PATH = "intel.yaml"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logger
configure_logger()
logger = logging.getLogger("Model Optimization")


class ModelOptimizer:
    def __init__(self, intel_path: str = INTEL_PATH, config_overrides: dict = None):
        self.intel_path = intel_path
        self.intel_config = self._load_intel()
        if config_overrides:
            self.intel_config.update(config_overrides)
        self.dataset_name = self.intel_config.get("dataset_name")
        self.model_name = self.intel_config.get("model_name")
        self.target_column = self.intel_config.get("target_column")

        # Store the logger as an instance attribute
        self.logger = logger

        # Load the data paths
        self.train_path = self.intel_config.get("train_transformed_path")
        self.test_path = self.intel_config.get("test_transformed_path")

        # Set up paths for saving outputs
        self.optimized_model_dir = os.path.join(ROOT_DIR, "model", f"model_{self.dataset_name}")
        self.optimized_model_path = os.path.join(self.optimized_model_dir, "optimized_model.pkl")

        self.best_params_dir = os.path.join(ROOT_DIR, "reports", "metrics", f"best_params_{self.dataset_name}")
        self.best_params_path = os.path.join(self.best_params_dir, "params.json")

        # Create directories if they don't exist
        os.makedirs(self.optimized_model_dir, exist_ok=True)
        os.makedirs(self.best_params_dir, exist_ok=True)

        # Load data
        self.X_train, self.y_train = self._load_data(self.train_path)
        self.X_test, self.y_test = self._load_data(self.test_path)

        # Available model and their hyperparameter spaces
        self.models = self._get_available_models()
        self.param_spaces = self._get_hyperparameter_spaces()

    def _load_intel(self) -> Dict[str, Any]:
        try:
            with open(self.intel_path, "r") as file:
                config = yaml.safe_load(file)
            logger.info(f"Successfully loaded configuration from {self.intel_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def _load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Successfully loaded data from {data_path}")

            # Separate features and target
            X = data.drop(columns=[self.target_column], errors='ignore')
            if self.target_column in data.columns:
                y = data[self.target_column]
            else:
                logger.error(f"Target column '{self.target_column}' not found in data")
                raise ValueError(f"Target column '{self.target_column}' not found in data")

            logger.info(f"Data shape - X: {X.shape}, y: {y.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _get_available_models(self) -> Dict[str, Any]:
        models = {
            "Logistic Regression": LogisticRegression,
            "Ridge Classifier": RidgeClassifier,
            "SGD Classifier": SGDClassifier,
            "Passive Aggressive": PassiveAggressiveClassifier,
            "Decision Tree": DecisionTreeClassifier,
            "Random Forest": RandomForestClassifier,
            "Gradient Boosting": GradientBoostingClassifier,
            "AdaBoost": AdaBoostClassifier,
            "Extra Trees": ExtraTreesClassifier,
            "K-Nearest Neighbors": KNeighborsClassifier,
            "Support Vector Classifier": SVC,
            "MLP Classifier": MLPClassifier,
            "Gaussian Naive Bayes": GaussianNB,
            "Multinomial Naive Bayes": MultinomialNB,
            "Bernoulli Naive Bayes": BernoulliNB,
            "Linear Discriminant Analysis": LinearDiscriminantAnalysis,
            "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis,
            "XGBoost": xgb.XGBClassifier,
            "LightGBM": lgb.LGBMClassifier,
            "CatBoost": cb.CatBoostClassifier
        }

        # Log available model
        section("Available Classification Models", logger)
        descriptions = {
            "Logistic Regression": "Logistic regression for binary and multiclass classification",
            "Ridge Classifier": "Ridge classifier with L2 regularization",
            "SGD Classifier": "Linear classifier fitted by minimizing a regularized loss function with SGD",
            "Passive Aggressive": "Passive Aggressive Classifier",
            "Decision Tree": "Decision tree classifier",
            "Random Forest": "Ensemble of decision trees using bootstrap sampling",
            "Gradient Boosting": "Gradient boosting for classification",
            "AdaBoost": "AdaBoost classification algorithm",
            "Extra Trees": "Extremely randomized trees for classification",
            "K-Nearest Neighbors": "Classification based on k-nearest neighbors",
            "Support Vector Classifier": "Support vector classification",
            "MLP Classifier": "Multi-layer Perceptron classifier",
            "Gaussian Naive Bayes": "Gaussian Naive Bayes classifier",
            "Multinomial Naive Bayes": "Multinomial Naive Bayes classifier",
            "Bernoulli Naive Bayes": "Bernoulli Naive Bayes classifier",
            "Linear Discriminant Analysis": "Linear Discriminant Analysis",
            "Quadratic Discriminant Analysis": "Quadratic Discriminant Analysis",
            "XGBoost": "XGBoost classification algorithm",
            "LightGBM": "LightGBM classification algorithm",
            "CatBoost": "CatBoost classification algorithm"
        }
        for i, (name, _) in enumerate(models.items(), 1):
            logger.info(f"{i}. {name} - {descriptions.get(name, '')}")

        return models

    def _get_hyperparameter_spaces(self) -> Dict[str, Dict[str, Any]]:
        param_spaces = {
            "Logistic Regression": {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l1", "l2", "elasticnet", "none"],
                "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                "max_iter": [100, 200, 500]
            },
            "Ridge Classifier": {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
            },
            "SGD Classifier": {
                "loss": ["hinge", "log_loss", "perceptron", "squared_hinge"],
                "penalty": ["l2", "l1", "elasticnet"],
                "alpha": [0.0001, 0.001, 0.01],
                "l1_ratio": [0.15, 0.5, 0.85]
            },
            "Passive Aggressive": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "loss": ["hinge", "squared_hinge"]
            },
            "Decision Tree": {
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["auto", "sqrt", "log2", None],
                "criterion": ["gini", "entropy"]
            },
            "Random Forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["auto", "sqrt", "log2"],
                "criterion": ["gini", "entropy"]
            },
            "Gradient Boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "subsample": [0.8, 0.9, 1.0]
            },
            "AdaBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 1.0],
                "algorithm": ["SAMME", "SAMME.R"]
            },
            "Extra Trees": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["auto", "sqrt", "log2"],
                "criterion": ["gini", "entropy"]
            },
            "K-Nearest Neighbors": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "p": [1, 2]
            },
            "Support Vector Classifier": {
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
                "probability": [True]  # Enable probability estimates for ROC-AUC
            },
            "MLP Classifier": {
                "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                "activation": ["relu", "tanh", "logistic"],
                "solver": ["adam", "sgd"],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate": ["constant", "adaptive"]
            },
            "Gaussian Naive Bayes": {
                "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
            },
            "Multinomial Naive Bayes": {
                "alpha": [0.1, 0.5, 1.0, 2.0]
            },
            "Bernoulli Naive Bayes": {
                "alpha": [0.1, 0.5, 1.0, 2.0],
                "binarize": [0.0, 0.1, 0.5]
            },
            "Linear Discriminant Analysis": {
                "solver": ["svd", "lsqr", "eigen"]
            },
            "Quadratic Discriminant Analysis": {
                "reg_param": [0.0, 0.01, 0.1]
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "gamma": [1e-5, 0.1, 0.2],
                "objective": ["binary:logistic"]  # Will need to handle multiclass separately
            },
            "LightGBM": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7, -1],
                "num_leaves": [31, 50, 100],
                "min_child_samples": [20, 50, 100],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "objective": ["binary"]  # Will need to handle multiclass separately
            },
            "CatBoost": {
                "iterations": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "depth": [4, 6, 8],
                "l2_leaf_reg": [1, 3, 5, 7],
                "border_count": [32, 64, 128],
                "verbose": [False]
            }
        }
        return param_spaces

    def _get_model_class(self) -> Any:
        for model_key, model_class in self.models.items():
            if model_key.lower().replace(" ", "") == self.model_name.lower().replace(" ", ""):
                logger.info(f"Found model class for '{self.model_name}': {model_key}")
                return model_class

        # Direct mapping if the above fails
        model_mapping = {
            "LogisticRegression": LogisticRegression,
            "RidgeClassifier": RidgeClassifier,
            "SGDClassifier": SGDClassifier,
            "PassiveAggressiveClassifier": PassiveAggressiveClassifier,
            "DecisionTree": DecisionTreeClassifier,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "RandomForest": RandomForestClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "GradientBoosting": GradientBoostingClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "AdaBoost": AdaBoostClassifier,
            "AdaBoostClassifier": AdaBoostClassifier,
            "ExtraTrees": ExtraTreesClassifier,
            "ExtraTreesClassifier": ExtraTreesClassifier,
            "KNN": KNeighborsClassifier,
            "KNeighborsClassifier": KNeighborsClassifier,
            "SVC": SVC,
            "SupportVectorClassifier": SVC,
            "MLP": MLPClassifier,
            "MLPClassifier": MLPClassifier,
            "GaussianNB": GaussianNB,
            "MultinomialNB": MultinomialNB,
            "BernoulliNB": BernoulliNB,
            "LDA": LinearDiscriminantAnalysis,
            "QDA": QuadraticDiscriminantAnalysis,
            "XGBoost": xgb.XGBClassifier,
            "XGBClassifier": xgb.XGBClassifier,
            "LightGBM": lgb.LGBMClassifier,
            "LGBMClassifier": lgb.LGBMClassifier,
            "CatBoost": cb.CatBoostClassifier,
            "CatBoostClassifier": cb.CatBoostClassifier
        }

        if self.model_name in model_mapping:
            logger.info(f"Found model class for '{self.model_name}' in direct mapping")
            return model_mapping[self.model_name]

        logger.error(f"Model '{self.model_name}' not found in available model")
        raise ValueError(f"Model '{self.model_name}' not found in available model")

    def _get_param_space(self) -> Dict[str, Any]:
        for model_key, param_space in self.param_spaces.items():
            if model_key.lower().replace(" ", "") == self.model_name.lower().replace(" ", ""):
                return param_space

        # Try additional mappings
        model_mapping = {
            "LinearRegression": "Linear Regression",
            "Ridge": "Ridge Regression",
            "RidgeRegression": "Ridge Regression",
            "Lasso": "Lasso Regression",
            "LassoRegression": "Lasso Regression",
            "ElasticNet": "ElasticNet",
            "SGDRegressor": "SGD Regressor",
            "BayesianRidge": "Bayesian Ridge",
            "HuberRegressor": "Huber Regressor",
            "RANSACRegressor": "RANSAC Regressor",
            "DecisionTree": "Decision Tree",
            "DecisionTreeRegressor": "Decision Tree",
            "RandomForest": "Random Forest",
            "RandomForestRegressor": "Random Forest",
            "GradientBoosting": "Gradient Boosting",
            "GradientBoostingRegressor": "Gradient Boosting",
            "AdaBoost": "AdaBoost",
            "AdaBoostRegressor": "AdaBoost",
            "ExtraTrees": "Extra Trees",
            "ExtraTreesRegressor": "Extra Trees",
            "KNN": "K-Nearest Neighbors",
            "KNeighborsRegressor": "K-Nearest Neighbors",
            "SVR": "Support Vector Regression",
            "SupportVectorRegression": "Support Vector Regression",
            "MLP": "MLP Regressor",
            "MLPRegressor": "MLP Regressor",
            "XGBoost": "XGBoost",
            "XGBRegressor": "XGBoost",
            "LightGBM": "LightGBM",
            "LGBMRegressor": "LightGBM",
            "CatBoost": "CatBoost",
            "CatBoostRegressor": "CatBoost"
        }

        if self.model_name in model_mapping:
            mapped_name = model_mapping[self.model_name]
            if mapped_name in self.param_spaces:
                return self.param_spaces[mapped_name]

        logger.error(f"Parameter space for model '{self.model_name}' not found")
        raise ValueError(f"Parameter space for model '{self.model_name}' not found")

    def _calculate_metric(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            metric_name: str,
            y_pred_proba: Optional[np.ndarray] = None
    ) -> float:
        if metric_name == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif metric_name == "precision":
            return precision_score(y_true, y_pred, average='weighted', zero_division=0)
        elif metric_name == "recall":
            return recall_score(y_true, y_pred, average='weighted', zero_division=0)
        elif metric_name == "f1_score":
            return f1_score(y_true, y_pred, average='weighted', zero_division=0)
        elif metric_name == "roc_auc":
            if y_pred_proba is not None:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    return roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # Multiclass
                    return roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            else:
                raise ValueError("ROC-AUC requires probability predictions")
        elif metric_name == "log_loss":
            if y_pred_proba is not None:
                return log_loss(y_true, y_pred_proba)
            else:
                raise ValueError("Log loss requires probability predictions")
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def optimize_with_grid_search(
            self,
            cv: int = 5,
            scoring: str = "accuracy",  # Changed from "neg_mean_squared_error"
            n_jobs: int = -1
    ) -> Tuple[Any, Dict[str, Any]]:
        section("Grid Search Optimization", logger)

        model_class = self._get_model_class()
        param_grid = self._get_param_space()

        logger.info(f"Starting grid search for {self.model_name}")
        logger.info(f"Hyperparameter grid: {param_grid}")

        model = model_class()
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )

        logger.info("Fitting grid search...")
        grid_search.fit(self.X_train, self.y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        logger.info(f"Grid search complete. Best score: {best_score}")
        logger.info(f"Best parameters: {best_params}")

        return best_model, best_params

    def _objective(
            self,
            trial: optuna.Trial,
            model_class: Any,
            param_space: Dict[str, Any],
            metric_name: str,
            maximize: bool
    ) -> float:
        params = {}
        for param_name, param_values in param_space.items():
            if isinstance(param_values, list):
                if all(isinstance(val, (int, float)) for val in param_values) and len(param_values) > 1:
                    if all(isinstance(val, int) for val in param_values):
                        params[param_name] = trial.suggest_int(
                            param_name,
                            min(param_values),
                            max(param_values),
                            log=max(param_values) / max(1, min(param_values)) > 100
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name,
                            min(param_values),
                            max(param_values),
                            log=max(param_values) / max(1e-10, min(param_values)) > 100
                        )
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)

        if "hidden_layer_sizes" in param_space:
            params["hidden_layer_sizes"] = trial.suggest_categorical("hidden_layer_sizes",
                                                                     param_space["hidden_layer_sizes"])

        if model_class.__name__ == "CatBoostClassifier":
            params["verbose"] = False  # Force silent mode
            params["allow_writing_files"] = False  # Disable log files
            params["thread_count"] = 1  # Prevent threading issues

        model = model_class(**params)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        # Get probability predictions if needed for certain metrics
        y_pred_proba = None
        if metric_name in ["roc_auc", "log_loss"] and hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(self.X_test)

        metric_value = self._calculate_metric(self.y_test, y_pred, metric_name, y_pred_proba)

        logger.info(
            f"Trial {trial.number} - Params: {params}, "
            f"Metric ({metric_name}): {metric_value:.4f}"
        )

        return metric_value if maximize else -metric_value

    def optimize_with_optuna(
            self,
            n_trials: int,
            metric_name: str = "rmse",
            maximize: bool = False
    ) -> Tuple[Any, Dict[str, Any]]:
        section("Optuna Optimization", logger)

        model_class = self._get_model_class()
        param_space = self._get_param_space()

        logger.info(f"Starting Optuna optimization for {self.model_name}")
        logger.info(f"Number of trials: {n_trials}")
        logger.info(f"Metric to {'maximize' if maximize else 'minimize'}: {metric_name}")

        direction = "maximize" if maximize else "minimize"
        study = optuna.create_study(direction=direction)

        study.optimize(
            lambda trial: self._objective(trial, model_class, param_space, metric_name, maximize),
            n_trials=n_trials
        )

        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value

        if maximize:
            logger.info(f"Optimization complete. Best {metric_name}: {best_value:.4f}")
        else:
            logger.info(f"Optimization complete. Best {metric_name}: {-best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        best_model = model_class(**best_params)
        best_model.fit(self.X_train, self.y_train)

        return best_model, best_params

    def save_optimized_model(self, model: Any, best_params: Dict[str, Any]) -> None:
        section("Saving Optimized Model", logger)

        try:
            # FIX: Open file in binary write mode instead of passing path string
            with open(self.optimized_model_path, 'wb') as file:
                cloudpickle.dump(model, file)
            logger.info(f"Optimized model saved to {self.optimized_model_path}")
        except Exception as e:
            logger.error(f"Error saving optimized model: {str(e)}")
            raise

        try:
            with open(self.best_params_path, "w") as file:
                yaml.dump(best_params, file)
            logger.info(f"Best parameters saved to {self.best_params_path}")
        except Exception as e:
            logger.error(f"Error saving best parameters: {str(e)}")
            raise

    def update_intel_yaml(self) -> None:
        section("Updating Intel YAML", logger)

        try:
            self.intel_config["optimized_model_path"] = self.optimized_model_path
            self.intel_config["best_params_path"] = self.best_params_path
            self.intel_config["optimization_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(self.intel_path, "w") as file:
                yaml.dump(self.intel_config, file)

            logger.info(f"Updated intel.yaml with optimized model information")
        except Exception as e:
            logger.error(f"Error updating intel.yaml: {str(e)}")
            raise

def reset_catboost_logging():
    while _custom_loggers_stack:
        try:
            _custom_loggers_stack.pop()
        except IndexError:
            break

atexit.register(reset_catboost_logging)

def get_available_metrics():
    return {
        "1": ("accuracy", True, "Accuracy Score"),
        "2": ("f1_score", True, "F1 Score (Weighted)"),
        "3": ("precision", True, "Precision (Weighted)"),
        "4": ("recall", True, "Recall (Weighted)"),
        "5": ("roc_auc", True, "ROC-AUC Score"),
        "6": ("log_loss", False, "Log Loss")
    }


def get_optimization_methods():
    return {
        "1": "Grid Search",
        "2": "Optuna"
    }


def optimize_model(
        optimize: bool = True,
        method: str = "1",
        n_trials: int = 50,
        metric: str = "1",
        config_overrides: dict = None
) -> dict:
    result = {
        "status": "success",
        "message": "",
        "best_params": None,
        "model_path": None,
        "metrics": {}
    }

    try:
        if not optimize:
            result["message"] = "Optimization skipped by user choice"
            return result

        logger.info("Starting model optimization process")
        optimizer = ModelOptimizer(config_overrides=config_overrides)

        if method == "1":
            optimized_model, best_params = optimizer.optimize_with_grid_search()
        else:
            metric_mapping = get_available_metrics()

            if metric not in metric_mapping:
                logger.warning("Invalid metric choice, defaulting to Accuracy")
                metric = "1"

            metric_name, maximize, _ = metric_mapping[metric]
            optimized_model, best_params = optimizer.optimize_with_optuna(
                n_trials=n_trials,
                metric_name=metric_name,
                maximize=maximize
            )

        optimizer.save_optimized_model(optimized_model, best_params)
        optimizer.update_intel_yaml()

        # Generate predictions
        y_pred = optimized_model.predict(optimizer.X_test)
        y_pred_proba = None

        # Get probability predictions if the model supports it
        if hasattr(optimized_model, "predict_proba"):
            try:
                y_pred_proba = optimized_model.predict_proba(optimizer.X_test)
            except Exception as e:
                logger.warning(f"Could not get probability predictions: {str(e)}")

        # Calculate classification metrics
        metrics = {
            "accuracy": accuracy_score(optimizer.y_test, y_pred),
            "f1_score": f1_score(optimizer.y_test, y_pred, average='weighted', zero_division=0),
            "precision": precision_score(optimizer.y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(optimizer.y_test, y_pred, average='weighted', zero_division=0)
        }

        # Add ROC-AUC if probability predictions are available
        if y_pred_proba is not None:
            try:
                unique_classes = len(np.unique(optimizer.y_test))
                if unique_classes == 2:  # Binary classification
                    metrics["roc_auc"] = roc_auc_score(optimizer.y_test, y_pred_proba[:, 1])
                elif unique_classes > 2:  # Multiclass classification
                    metrics["roc_auc"] = roc_auc_score(
                        optimizer.y_test,
                        y_pred_proba,
                        multi_class='ovr',
                        average='weighted'
                    )

                # Add log loss
                metrics["log_loss"] = log_loss(optimizer.y_test, y_pred_proba)
            except ValueError as e:
                logger.warning(f"Could not calculate ROC-AUC or Log Loss: {str(e)}")
            except Exception as e:
                logger.warning(f"Error calculating probability-based metrics: {str(e)}")

        # Log the final metrics
        logger.info("Final Model Performance:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name.upper()}: {metric_value:.4f}")

        result.update({
            "best_params": best_params,
            "model_path": optimizer.optimized_model_path,
            "metrics": metrics,
            "message": "Optimization completed successfully"
        })

    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        result.update({
            "status": "error",
            "message": str(e)
        })

    return result