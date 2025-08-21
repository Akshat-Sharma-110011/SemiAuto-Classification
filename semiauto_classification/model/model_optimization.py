import os
import yaml
import cloudpickle
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any, Tuple, List, Optional, Union
import json
import warnings

# Import optimization libraries
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna
import atexit
from catboost.core import _custom_loggers_stack

# Import classification metrics
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

# Import all model classes for ensemble optimization
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, VotingClassifier, BaggingClassifier, StackingClassifier
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
logger = logging.getLogger("Enhanced Model Optimization")

# Suppress common sklearn warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')


class EnsembleOptimizer:
    """Specialized optimizer for ensemble models"""

    def __init__(self, model, model_type: str):
        self.model = model
        self.model_type = model_type
        self.logger = logger

    def get_ensemble_param_space(self) -> Dict[str, Any]:
        """Get parameter space for ensemble models"""

        if isinstance(self.model, VotingClassifier):
            return self._get_voting_param_space()
        elif isinstance(self.model, StackingClassifier):
            return self._get_stacking_param_space()
        elif isinstance(self.model, BaggingClassifier):
            return self._get_bagging_param_space()
        else:
            return {}

    def _get_voting_param_space(self) -> Dict[str, Any]:
        """Parameter space for voting classifier"""
        return {
            'voting': ['soft', 'hard'],
            'flatten_transform': [True, False]  # Fixed: Use actual boolean values
        }

    def _get_stacking_param_space(self) -> Dict[str, Any]:
        """Parameter space for stacking classifier"""
        param_space = {
            'cv': [3, 5, 7],
            'stack_method': ['auto', 'predict_proba'],
            'passthrough': [True, False]  # Fixed: Use actual boolean values
        }

        # Add final estimator parameters if it's a simple model
        final_estimator = self.model.final_estimator
        if hasattr(final_estimator, 'get_params'):
            final_params = self._get_base_model_params(final_estimator)
            for param, values in final_params.items():
                param_space[f'final_estimator__{param}'] = values

        return param_space

    def _get_bagging_param_space(self) -> Dict[str, Any]:
        """Parameter space for bagging classifier"""
        param_space = {
            'n_estimators': [10, 20, 50, 100],
            'max_samples': [0.5, 0.7, 0.9, 1.0],
            'max_features': [0.5, 0.7, 0.9, 1.0],
            'bootstrap': [True, False],  # Fixed: Use actual boolean values
            'bootstrap_features': [True, False]  # Fixed: Use actual boolean values
        }

        # Add base estimator parameters
        base_estimator = self.model.base_estimator
        if hasattr(base_estimator, 'get_params') and base_estimator is not None:
            base_params = self._get_base_model_params(base_estimator)
            for param, values in base_params.items():
                param_space[f'base_estimator__{param}'] = values

        return param_space

    def _get_base_model_params(self, model) -> Dict[str, Any]:
        """Get parameter space for base models in ensemble"""
        model_name = model.__class__.__name__

        base_params = {
            'LogisticRegression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],  # Removed l1 to avoid solver conflicts
                'solver': ['lbfgs'],  # Use compatible solver
                'max_iter': [100, 200, 500]
            },
            'DecisionTreeClassifier': {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'RandomForestClassifier': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            },
            'SVC': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }

        return base_params.get(model_name, {})


class EnhancedModelOptimizer:
    """Enhanced optimizer supporting both baseline and ensemble models"""

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
        self.train_path = self.intel_config.get("train_selected_path")
        self.test_path = self.intel_config.get("test_selected_path")

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

        # Available models and their hyperparameter spaces
        self.models = self._get_available_models()
        self.param_spaces = self._get_hyperparameter_spaces()

        # Load current model
        self.current_model = self._load_current_model()
        self.model_type = self._identify_model_type()

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

    def _load_current_model(self):
        """Load the current model for optimization"""
        try:
            model_path = self.intel_config.get('model_path')
            if model_path and os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = cloudpickle.load(f)
                logger.info(f"Loaded current model from {model_path}")
                return model
            else:
                logger.error("No current model found to optimize")
                raise ValueError("No current model found to optimize")
        except Exception as e:
            logger.error(f"Error loading current model: {str(e)}")
            raise

    def _identify_model_type(self) -> str:
        """Identify the type of model (baseline or ensemble)"""
        if isinstance(self.current_model, (VotingClassifier, StackingClassifier, BaggingClassifier)):
            return "ensemble"
        else:
            return "baseline"

    def _get_available_models(self) -> Dict[str, Any]:
        """Get available model classes"""
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
        return models

    def _get_hyperparameter_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Get updated hyperparameter spaces for all models (fixed deprecated parameters)"""
        param_spaces = {
            "Logistic Regression": {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l2"],  # Removed problematic penalties
                "solver": ["lbfgs", "newton-cg"],  # Compatible solvers
                "max_iter": [100, 200, 500]
            },
            "Ridge Classifier": {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                "solver": ["auto", "svd", "cholesky", "lsqr"]
            },
            "SGD Classifier": {
                "loss": ["hinge", "log_loss", "perceptron"],
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
                "max_features": [None, "sqrt", "log2"],  # Removed "auto"
                "criterion": ["gini", "entropy"]
            },
            "Random Forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],  # Removed "auto"
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
                "learning_rate": [0.01, 0.1, 1.0]
                # Removed deprecated 'algorithm' parameter
            },
            "Extra Trees": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],  # Removed "auto"
                "criterion": ["gini", "entropy"]
            },
            "K-Nearest Neighbors": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "p": [1, 2]
            },
            "Support Vector Classifier": {
                "kernel": ["linear", "poly", "rbf"],
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
                "probability": [True]  # Already boolean
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
                "reg_param": [0.01, 0.1, 0.5]  # Increased minimum to avoid rank issues
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "gamma": [0, 0.1, 0.2],
                "objective": ["binary:logistic"]
            },
            "LightGBM": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7, -1],
                "num_leaves": [31, 50, 100],
                "min_child_samples": [20, 50, 100],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "objective": ["binary"]
            },
            "CatBoost": {
                "iterations": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "depth": [4, 6, 8],
                "l2_leaf_reg": [1, 3, 5, 7],
                "border_count": [32, 64, 128],
                "verbose": [False]  # Already boolean
            }
        }
        return param_spaces

    def _get_param_space(self) -> Dict[str, Any]:
        """Get parameter space for the current model"""
        if self.model_type == "ensemble":
            # Use ensemble optimizer
            ensemble_optimizer = EnsembleOptimizer(self.current_model, self.model_type)
            return ensemble_optimizer.get_ensemble_param_space()
        else:
            # Use baseline model parameter space
            model_class_name = self.current_model.__class__.__name__

            # Map class names to our parameter spaces
            for model_key, param_space in self.param_spaces.items():
                if model_key.lower().replace(" ", "") == model_class_name.lower().replace("classifier", ""):
                    return param_space

            # If no direct match, try to infer from class name
            if hasattr(self.current_model, 'get_params'):
                # Generate a basic parameter space based on current parameters
                current_params = self.current_model.get_params()
                basic_param_space = {}

                for param_name, param_value in current_params.items():
                    if isinstance(param_value, (int, float)):
                        if param_name in ['n_estimators', 'max_iter', 'n_neighbors']:
                            basic_param_space[param_name] = [max(1, int(param_value * 0.5)),
                                                             param_value,
                                                             int(param_value * 1.5)]
                        elif param_name in ['learning_rate', 'alpha', 'C']:
                            basic_param_space[param_name] = [param_value * 0.1,
                                                             param_value,
                                                             param_value * 10]

                return basic_param_space

            logger.warning(f"No parameter space found for {model_class_name}")
            return {}

    def _calculate_metric(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            metric_name: str,
            y_pred_proba: Optional[np.ndarray] = None
    ) -> float:
        """Calculate specified metric"""
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
            scoring: str = "accuracy",
            n_jobs: int = -1
    ) -> Tuple[Any, Dict[str, Any]]:
        """Grid search optimization for any model type"""
        section(f"Grid Search Optimization for {self.model_type} model", logger)

        param_grid = self._get_param_space()

        if not param_grid:
            logger.warning("No parameters to optimize")
            return self.current_model, {}

        logger.info(f"Starting grid search optimization")
        logger.info(f"Hyperparameter grid: {param_grid}")

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            grid_search = GridSearchCV(
                estimator=self.current_model,
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

    def _suggest_parameter(self, trial: optuna.Trial, param_name: str, param_values: Any) -> Any:
        """Suggest parameter value with proper type handling - FIXED VERSION"""
        if isinstance(param_values, list):
            # Special handling for boolean parameters
            if all(isinstance(val, bool) for val in param_values):
                return trial.suggest_categorical(param_name, param_values)

            # Check if all values are numeric
            elif all(isinstance(val, (int, float)) for val in param_values) and len(param_values) > 1:
                min_val, max_val = min(param_values), max(param_values)

                # Check if all values are integers
                if all(isinstance(val, int) for val in param_values):
                    return trial.suggest_int(
                        param_name,
                        min_val,
                        max_val,
                        log=max_val / max(1, min_val) > 100
                    )
                else:
                    return trial.suggest_float(
                        param_name,
                        min_val,
                        max_val,
                        log=max_val / max(1e-10, min_val) > 100
                    )
            else:
                # Categorical parameter
                return trial.suggest_categorical(param_name, param_values)
        else:
            # Single value or non-list parameter
            return param_values

    def _objective(
            self,
            trial: optuna.Trial,
            param_space: Dict[str, Any],
            metric_name: str,
            maximize: bool
    ) -> float:
        """Optuna objective function with improved parameter handling"""
        params = {}

        # Generate parameters for this trial
        for param_name, param_values in param_space.items():
            params[param_name] = self._suggest_parameter(trial, param_name, param_values)

        # Create model with suggested parameters
        try:
            if self.model_type == "ensemble":
                # Clone the ensemble model and update parameters
                model = self._clone_ensemble_with_params(params)
            else:
                # Create new instance of baseline model with parameters
                model_class = self.current_model.__class__
                current_params = self.current_model.get_params()
                current_params.update(params)

                # Handle CatBoost specific parameters
                if model_class.__name__ == "CatBoostClassifier":
                    current_params["verbose"] = False
                    current_params["allow_writing_files"] = False
                    current_params["thread_count"] = 1

                model = model_class(**current_params)

            # Train and evaluate with suppressed warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

                # Get probability predictions if needed
                y_pred_proba = None
                if metric_name in ["roc_auc", "log_loss"] and hasattr(model, "predict_proba"):
                    try:
                        y_pred_proba = model.predict_proba(self.X_test)
                    except:
                        pass

                metric_value = self._calculate_metric(self.y_test, y_pred, metric_name, y_pred_proba)

            logger.info(
                f"Trial {trial.number} - Params: {params}, "
                f"Metric ({metric_name}): {metric_value:.4f}"
            )

            return metric_value if maximize else -metric_value

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {str(e)}")
            return -float('inf') if maximize else float('inf')

    def _clone_ensemble_with_params(self, params: Dict[str, Any]):
        """Clone ensemble model with new parameters - FIXED VERSION"""
        if isinstance(self.current_model, VotingClassifier):
            # For voting classifier, update top-level parameters
            current_params = self.current_model.get_params()

            # Filter parameters to only include valid ones and ensure proper types
            valid_params = {}
            for k, v in params.items():
                if k in ['voting', 'flatten_transform']:
                    # Ensure boolean parameters are actually boolean
                    if k == 'flatten_transform' and isinstance(v, (int, float)):
                        valid_params[k] = bool(v)
                    else:
                        valid_params[k] = v

            return VotingClassifier(
                estimators=self.current_model.estimators,
                **valid_params
            )
        elif isinstance(self.current_model, StackingClassifier):
            # For stacking classifier, handle nested parameters
            final_estimator_params = {}
            stacking_params = {}

            for k, v in params.items():
                if k.startswith('final_estimator__'):
                    final_estimator_params[k.replace('final_estimator__', '')] = v
                elif k in ['cv', 'stack_method', 'passthrough']:
                    # Ensure boolean parameters are actually boolean
                    if k == 'passthrough' and isinstance(v, (int, float)):
                        stacking_params[k] = bool(v)
                    else:
                        stacking_params[k] = v

            # Update final estimator if needed
            final_estimator = self.current_model.final_estimator
            if final_estimator_params and final_estimator is not None:
                final_estimator_class = final_estimator.__class__
                current_final_params = final_estimator.get_params()
                current_final_params.update(final_estimator_params)
                final_estimator = final_estimator_class(**current_final_params)

            return StackingClassifier(
                estimators=self.current_model.estimators,
                final_estimator=final_estimator,
                **stacking_params
            )
        elif isinstance(self.current_model, BaggingClassifier):
            # For bagging classifier, handle base estimator parameters
            base_estimator_params = {}
            bagging_params = {}

            for k, v in params.items():
                if k.startswith('base_estimator__'):
                    base_estimator_params[k.replace('base_estimator__', '')] = v
                elif k in ['n_estimators', 'max_samples', 'max_features', 'bootstrap', 'bootstrap_features']:
                    # Ensure boolean parameters are actually boolean
                    if k in ['bootstrap', 'bootstrap_features'] and isinstance(v, (int, float)):
                        bagging_params[k] = bool(v)
                    else:
                        bagging_params[k] = v

            # Update base estimator if needed
            base_estimator = self.current_model.base_estimator
            if base_estimator_params and base_estimator is not None:
                base_estimator_class = base_estimator.__class__
                current_base_params = base_estimator.get_params()
                current_base_params.update(base_estimator_params)
                base_estimator = base_estimator_class(**current_base_params)

            return BaggingClassifier(
                base_estimator=base_estimator,
                **bagging_params
            )
        else:
            # Fallback: return original model
            return self.current_model

    def optimize_with_optuna(
            self,
            n_trials: int,
            metric_name: str = "accuracy",
            maximize: bool = True
    ) -> Tuple[Any, Dict[str, Any]]:
        """Optuna optimization for any model type"""
        section(f"Optuna Optimization for {self.model_type} model", logger)

        param_space = self._get_param_space()

        if not param_space:
            logger.warning("No parameters to optimize")
            return self.current_model, {}

        logger.info(f"Starting Optuna optimization")
        logger.info(f"Number of trials: {n_trials}")
        logger.info(f"Metric to {'maximize' if maximize else 'minimize'}: {metric_name}")

        direction = "maximize" if maximize else "minimize"

        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction=direction)

        study.optimize(
            lambda trial: self._objective(trial, param_space, metric_name, maximize),
            n_trials=n_trials,
            show_progress_bar=False
        )

        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value

        if maximize:
            logger.info(f"Optimization complete. Best {metric_name}: {best_value:.4f}")
        else:
            logger.info(f"Optimization complete. Best {metric_name}: {-best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        # Create best model
        if self.model_type == "ensemble":
            best_model = self._clone_ensemble_with_params(best_params)
        else:
            model_class = self.current_model.__class__
            current_params = self.current_model.get_params()
            current_params.update(best_params)
            best_model = model_class(**current_params)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            best_model.fit(self.X_train, self.y_train)

        return best_model, best_params

    def save_optimized_model(self, model: Any, best_params: Dict[str, Any]) -> None:
        """Save optimized model and parameters"""
        section("Saving Optimized Model", logger)

        try:
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
        """Update intel YAML with optimization results"""
        section("Updating Intel YAML", logger)

        try:
            self.intel_config["optimized_model_path"] = self.optimized_model_path
            self.intel_config["best_params_path"] = self.best_params_path
            self.intel_config["optimization_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.intel_config["model_type"] = self.model_type

            with open(self.intel_path, "w") as file:
                yaml.dump(self.intel_config, file)

            logger.info(f"Updated intel.yaml with optimized model information")
        except Exception as e:
            logger.error(f"Error updating intel.yaml: {str(e)}")
            raise


def reset_catboost_logging():
    """Reset CatBoost logging"""
    while _custom_loggers_stack:
        try:
            _custom_loggers_stack.pop()
        except IndexError:
            break


atexit.register(reset_catboost_logging)


def get_available_metrics():
    """Get available optimization metrics"""
    return {
        "1": ("accuracy", True, "Accuracy Score"),
        "2": ("f1_score", True, "F1 Score (Weighted)"),
        "3": ("precision", True, "Precision (Weighted)"),
        "4": ("recall", True, "Recall (Weighted)"),
        "5": ("roc_auc", True, "ROC-AUC Score"),
        "6": ("log_loss", False, "Log Loss")
    }


def get_optimization_methods():
    """Get available optimization methods"""
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
    """Main optimization function supporting both baseline and ensemble models"""
    result = {
        "status": "success",
        "message": "",
        "best_params": None,
        "model_path": None,
        "metrics": {},
        "model_type": "unknown"
    }

    try:
        if not optimize:
            result["message"] = "Optimization skipped by user choice"
            return result

        logger.info("Starting enhanced model optimization process")

        # Suppress warnings during optimization
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            optimizer = EnhancedModelOptimizer(config_overrides=config_overrides)

            result["model_type"] = optimizer.model_type
            logger.info(f"Detected model type: {optimizer.model_type}")

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

            # Generate predictions for evaluation
            y_pred = optimized_model.predict(optimizer.X_test)
            y_pred_proba = None

            # Get probability predictions if the model supports it
            if hasattr(optimized_model, "predict_proba"):
                try:
                    y_pred_proba = optimized_model.predict_proba(optimizer.X_test)
                except Exception as e:
                    logger.warning(f"Could not get probability predictions: {str(e)}")

            # Calculate comprehensive metrics
            metrics = {
                "accuracy": accuracy_score(optimizer.y_test, y_pred),
                "f1_score": f1_score(optimizer.y_test, y_pred, average='weighted', zero_division=0),
                "precision": precision_score(optimizer.y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(optimizer.y_test, y_pred, average='weighted', zero_division=0)
            }

            # Add probability-based metrics if available
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
            logger.info(f"Final {optimizer.model_type.title()} Model Performance:")
            for metric_name, metric_value in metrics.items():
                logger.info(f"{metric_name.upper()}: {metric_value:.4f}")

            result.update({
                "best_params": best_params,
                "model_path": optimizer.optimized_model_path,
                "metrics": metrics,
                "message": f"{optimizer.model_type.title()} model optimization completed successfully"
            })

    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        result.update({
            "status": "error",
            "message": str(e)
        })

    return result


if __name__ == "__main__":
    print("Enhanced Model Optimization Script")
    print("Supports optimization of both baseline and ensemble models")

    # Example usage
    result = optimize_model(
        optimize=True,
        method="2",  # Optuna
        n_trials=30,
        metric="1"  # Accuracy
    )

    print(f"Optimization result: {result['status']}")
    print(f"Message: {result['message']}")
    if result['best_params']:
        print(f"Best parameters: {result['best_params']}")