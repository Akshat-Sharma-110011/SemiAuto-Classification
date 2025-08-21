#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import pandas as pd
import numpy as np
import cloudpickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
from sklearn.model_selection import cross_val_predict
import time
from sklearn.linear_model import (LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier, BaggingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

import sys
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from semiauto_classification.logger import get_logger, section, configure_logger

configure_logger()
logger = logging.getLogger("Enhanced Model Building")

class ModelParameterValidator:
    def __init__(self):
        self.parameter_types = {
            'LogisticRegression': {'C': float, 'max_iter': int, 'solver': str, 'random_state': int, 'n_jobs': int, 'penalty': str, 'tol': float, 'fit_intercept': bool, 'intercept_scaling': float, 'class_weight': (str, dict, type(None)), 'dual': bool, 'multi_class': str, 'warm_start': bool, 'l1_ratio': (float, type(None))},
            'RandomForestClassifier': {'n_estimators': int, 'max_depth': (int, type(None)), 'min_samples_split': int, 'min_samples_leaf': int, 'max_features': (str, int, float, type(None)), 'n_jobs': int, 'random_state': int, 'criterion': str, 'max_leaf_nodes': (int, type(None)), 'min_impurity_decrease': float, 'bootstrap': bool, 'oob_score': bool, 'warm_start': bool, 'class_weight': (str, dict, type(None)), 'ccp_alpha': float, 'max_samples': (int, float, type(None))},
            'XGBClassifier': {'n_estimators': int, 'learning_rate': float, 'max_depth': int, 'subsample': float, 'colsample_bytree': float, 'objective': str, 'n_jobs': int, 'eval_metric': str, 'random_state': int, 'verbosity': int, 'reg_alpha': float, 'reg_lambda': float, 'gamma': float, 'min_child_weight': int, 'colsample_bylevel': float, 'colsample_bynode': float, 'scale_pos_weight': float, 'base_score': float, 'booster': str, 'tree_method': str, 'num_class': int},
            'CatBoostClassifier': {'iterations': int, 'learning_rate': float, 'depth': int, 'l2_leaf_reg': float, 'loss_function': str, 'verbose': (bool, int), 'random_state': int, 'border_count': int, 'bagging_temperature': float, 'random_strength': float, 'one_hot_max_size': int, 'rsm': float, 'nan_mode': str, 'leaf_estimation_method': str, 'thread_count': int, 'od_type': str, 'od_wait': int},
            'LGBMClassifier': {'n_estimators': int, 'learning_rate': float, 'max_depth': int, 'num_leaves': int, 'subsample': float, 'colsample_bytree': float, 'objective': str, 'n_jobs': int, 'random_state': int, 'verbose': int, 'reg_alpha': float, 'reg_lambda': float, 'min_split_gain': float, 'min_child_weight': float, 'min_child_samples': int, 'subsample_freq': int, 'colsample_bylevel': float, 'max_bin': int, 'num_class': int},
        }

    def convert_value(self, value: str, target_type):
        if isinstance(target_type, tuple):
            if type(None) in target_type and (value is None or str(value).lower() in ['none', 'null', '']):
                return None
            for t in target_type:
                if t == type(None):
                    continue
                try:
                    return self.convert_value(value, t)
                except:
                    continue
            return value
        if target_type == bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ['true', '1', 'yes', 'on']
            return bool(value)
        elif target_type == int:
            if isinstance(value, int):
                return value
            return int(float(str(value)))
        elif target_type == float:
            if isinstance(value, (int, float)):
                return float(value)
            return float(str(value))
        elif target_type == str:
            return str(value)
        elif target_type == tuple:
            if isinstance(value, tuple):
                return value
            elif isinstance(value, str):
                try:
                    if value.startswith('(') and value.endswith(')'):
                        inner = value[1:-1].strip()
                        if ',' in inner:
                            return tuple(int(x.strip()) for x in inner.split(',') if x.strip())
                        elif inner:
                            return (int(inner),)
                        else:
                            return ()
                    else:
                        return (int(value),)
                except:
                    return (100,)
            elif isinstance(value, (list, int)):
                if isinstance(value, int):
                    return (value,)
                return tuple(value)
            else:
                return value
        else:
            return value

    def validate_and_convert_parameters(self, model_class_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if model_class_name not in self.parameter_types:
            return params
        expected_types = self.parameter_types[model_class_name]
        validated_params = {}
        for param_name, param_value in params.items():
            if param_name in expected_types:
                expected_type = expected_types[param_name]
                try:
                    validated_params[param_name] = self.convert_value(param_value, expected_type)
                except Exception as e:
                    print(f"Warning: Could not convert parameter {param_name}={param_value} to {expected_type}: {e}")
                    validated_params[param_name] = param_value
            else:
                validated_params[param_name] = param_value
        return validated_params

class StagedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, architecture: List[Dict], final_meta_model=None, cv=5, stage_method='stacking', combine_method='concatenate', passthrough_original=False, reduction_strategy='auto'):
        self.architecture = architecture
        self.final_meta_model = final_meta_model
        self.cv = cv
        self.stage_method = stage_method
        self.combine_method = combine_method
        self.passthrough_original = passthrough_original
        self.reduction_strategy = reduction_strategy
        self.stages_ = []
        self.stage_performances_ = []
        self.final_stage_ = None

    def _create_estimator_from_config(self, estimator_config: Dict, available_models: Dict):
        model_name = estimator_config['model']
        model_params = estimator_config.get('params', {})
        model_class = available_models[model_name]['class']
        return model_class(**model_params)

    def _select_models_for_next_stage(self, current_models: List, performances: List[float], target_count: int, strategy: Optional[str] = None) -> List[int]:
        if len(current_models) <= target_count:
            return list(range(len(current_models)))
        strategy = strategy or self.reduction_strategy
        if strategy == 'best_performers':
            sorted_indices = np.argsort(performances)[::-1]
            return sorted_indices[:target_count].tolist()
        elif strategy == 'diverse_selection':
            sorted_indices = np.argsort(performances)[::-1]
            selected = [sorted_indices[0]]
            remaining_count = target_count - 1
            step = max(1, len(sorted_indices) // remaining_count) if remaining_count > 0 else 1
            for i in range(1, len(sorted_indices), step):
                if len(selected) >= target_count:
                    break
                selected.append(sorted_indices[i])
            while len(selected) < target_count and len(selected) < len(sorted_indices):
                for idx in sorted_indices:
                    if idx not in selected:
                        selected.append(idx)
                        break
            return selected[:target_count]
        else:
            if target_count >= len(current_models) * 0.7:
                return self._select_models_for_next_stage(current_models, performances, target_count, 'best_performers')
            else:
                return self._select_models_for_next_stage(current_models, performances, target_count, 'diverse_selection')

    def fit(self, X, y, available_models):
        self.classes_ = np.unique(y)
        X_current = X.copy()
        X_original = X.copy() if self.passthrough_original else None
        current_models = None
        for stage_idx, stage_config in enumerate(self.architecture):
            logger.info(f"Fitting stage {stage_idx + 1}/{len(self.architecture)}")
            target_model_count = stage_config.get('model_count', len(stage_config.get('models', [])))
            stage_models = []
            stage_predictions = []
            stage_performances = []
            if stage_idx == 0:
                models_to_use = stage_config.get('models', [])
            else:
                if current_models is None:
                    raise ValueError(f"No models available for stage {stage_idx + 1}")
                selected_indices = self._select_models_for_next_stage(current_models, self.stage_performances_[-1] if self.stage_performances_ else [0.5] * len(current_models), target_model_count, self.reduction_strategy)
                models_to_use = [current_models[i] for i in selected_indices]
            for model_idx, model_config in enumerate(models_to_use):
                if isinstance(model_config, dict) and 'model' in model_config:
                    estimator = self._create_estimator_from_config(model_config, available_models)
                else:
                    estimator = model_config
                estimator.fit(X_current, y)
                stage_models.append(estimator)
                try:
                    if hasattr(estimator, 'predict_proba'):
                        cv_pred = cross_val_predict(estimator, X_current, y, cv=self.cv, method='predict_proba')
                        if cv_pred.shape[1] > 2:
                            stage_predictions.append(cv_pred)
                        else:
                            stage_predictions.append(cv_pred[:, 1:2])
                    else:
                        cv_pred = cross_val_predict(estimator, X_current, y, cv=self.cv)
                        stage_predictions.append(cv_pred.reshape(-1, 1))
                    if hasattr(estimator, 'predict_proba'):
                        pred_proba = estimator.predict_proba(X_current)
                        if pred_proba.shape[1] == 2:
                            performance = roc_auc_score(y, pred_proba[:, 1])
                        else:
                            performance = roc_auc_score(y, pred_proba, multi_class='ovr', average='weighted')
                    else:
                        pred = estimator.predict(X_current)
                        performance = accuracy_score(y, pred)
                    stage_performances.append(performance)
                except Exception as e:
                    logger.warning(f"Error in cross-validation for model {model_idx}: {e}")
                    pred = estimator.predict(X_current)
                    stage_predictions.append(pred.reshape(-1, 1))
                    stage_performances.append(accuracy_score(y, pred))
            self.stages_.append(stage_models)
            self.stage_performances_.append(stage_performances)
            current_models = stage_models
            if stage_idx < len(self.architecture) - 1:
                stage_features = np.hstack(stage_predictions)
                if self.combine_method == 'concatenate':
                    if self.passthrough_original and X_original is not None:
                        X_current = np.hstack([X_original, stage_features])
                    else:
                        X_current = stage_features
                elif self.combine_method == 'average':
                    if len(stage_predictions) > 1 and all(p.shape[1] == stage_predictions[0].shape[1] for p in stage_predictions):
                        stage_features = np.mean(stage_predictions, axis=0)
                        if self.passthrough_original and X_original is not None:
                            X_current = np.hstack([X_original, stage_features])
                        else:
                            X_current = stage_features
                    else:
                        X_current = np.hstack(stage_predictions)
                elif self.combine_method == 'weighted_average':
                    if len(stage_predictions) > 1 and len(stage_performances) == len(stage_predictions):
                        weights = np.array(stage_performances)
                        weights = weights / weights.sum()
                        if all(p.shape[1] == stage_predictions[0].shape[1] for p in stage_predictions):
                            stage_features = np.average(stage_predictions, axis=0, weights=weights)
                            if self.passthrough_original and X_original is not None:
                                X_current = np.hstack([X_original, stage_features])
                            else:
                                X_current = stage_features
                        else:
                            X_current = np.hstack(stage_predictions)
                    else:
                        X_current = np.hstack(stage_predictions)
                logger.info(f"Stage {stage_idx + 1} output shape: {X_current.shape}")
        if self.final_meta_model is not None:
            final_stage_predictions = []
            for estimator in self.stages_[-1]:
                try:
                    if hasattr(estimator, 'predict_proba'):
                        cv_pred = cross_val_predict(estimator, X_current, y, cv=self.cv, method='predict_proba')
                        if cv_pred.shape[1] > 2:
                            final_stage_predictions.append(cv_pred)
                        else:
                            final_stage_predictions.append(cv_pred[:, 1:2])
                    else:
                        cv_pred = cross_val_predict(estimator, X_current, y, cv=self.cv)
                        final_stage_predictions.append(cv_pred.reshape(-1, 1))
                except:
                    pred = estimator.predict(X_current)
                    final_stage_predictions.append(pred.reshape(-1, 1))
            final_features = np.hstack(final_stage_predictions)
            if self.passthrough_original and X_original is not None:
                final_features = np.hstack([X_original, final_features])
            self.final_stage_ = clone(self.final_meta_model)
            self.final_stage_.fit(final_features, y)
            logger.info(f"Final meta model fitted with input shape: {final_features.shape}")
        return self

    def predict(self, X):
        final_features = self.transform(X)
        if self.final_stage_ is not None:
            return self.final_stage_.predict(final_features)
        else:
            X_current = X.copy()
            X_original = X.copy() if self.passthrough_original else None
            for stage_idx, stage_models in enumerate(self.stages_[:-1]):
                stage_predictions = []
                for estimator in stage_models:
                    if hasattr(estimator, 'predict_proba'):
                        pred_proba = estimator.predict_proba(X_current)
                        if pred_proba.shape[1] > 2:
                            stage_predictions.append(pred_proba)
                        else:
                            stage_predictions.append(pred_proba[:, 1:2])
                    else:
                        pred = estimator.predict(X_current)
                        stage_predictions.append(pred.reshape(-1, 1))
                stage_features = np.hstack(stage_predictions)
                if self.combine_method == 'concatenate':
                    if self.passthrough_original and X_original is not None:
                        X_current = np.hstack([X_original, stage_features])
                    else:
                        X_current = stage_features
                elif self.combine_method == 'average':
                    if len(stage_predictions) > 1 and all(p.shape[1] == stage_predictions[0].shape[1] for p in stage_predictions):
                        stage_features = np.mean(stage_predictions, axis=0)
                        if self.passthrough_original and X_original is not None:
                            X_current = np.hstack([X_original, stage_features])
                        else:
                            X_current = stage_features
                    else:
                        X_current = np.hstack(stage_predictions)
                elif self.combine_method == 'weighted_average':
                    if (len(stage_predictions) > 1 and stage_idx < len(self.stage_performances_) and len(self.stage_performances_[stage_idx]) == len(stage_predictions)):
                        weights = np.array(self.stage_performances_[stage_idx])
                        weights = weights / weights.sum()
                        if all(p.shape[1] == stage_predictions[0].shape[1] for p in stage_predictions):
                            stage_features = np.average(stage_predictions, axis=0, weights=weights)
                            if self.passthrough_original and X_original is not None:
                                X_current = np.hstack([X_original, stage_features])
                            else:
                                X_current = stage_features
                        else:
                            X_current = np.hstack(stage_predictions)
                    else:
                        X_current = np.hstack(stage_predictions)
            predictions = []
            for estimator in self.stages_[-1]:
                predictions.append(estimator.predict(X_current))
            predictions = np.array(predictions).T
            return np.array([np.bincount(row).argmax() for row in predictions])

    def predict_proba(self, X):
        final_features = self.transform(X)
        if self.final_stage_ is not None and hasattr(self.final_stage_, 'predict_proba'):
            return self.final_stage_.predict_proba(final_features)
        else:
            X_current = X.copy()
            X_original = X.copy() if self.passthrough_original else None
            for stage_idx, stage_models in enumerate(self.stages_[:-1]):
                stage_predictions = []
                for estimator in stage_models:
                    if hasattr(estimator, 'predict_proba'):
                        pred_proba = estimator.predict_proba(X_current)
                        if pred_proba.shape[1] > 2:
                            stage_predictions.append(pred_proba)
                        else:
                            stage_predictions.append(pred_proba[:, 1:2])
                    else:
                        pred = estimator.predict(X_current)
                        stage_predictions.append(pred.reshape(-1, 1))
                stage_features = np.hstack(stage_predictions)
                if self.combine_method == 'concatenate':
                    if self.passthrough_original and X_original is not None:
                        X_current = np.hstack([X_original, stage_features])
                    else:
                        X_current = stage_features
                elif self.combine_method == 'average':
                    if len(stage_predictions) > 1 and all(p.shape[1] == stage_predictions[0].shape[1] for p in stage_predictions):
                        stage_features = np.mean(stage_predictions, axis=0)
                        if self.passthrough_original and X_original is not None:
                            X_current = np.hstack([X_original, stage_features])
                        else:
                            X_current = stage_features
                    else:
                        X_current = np.hstack(stage_predictions)
                elif self.combine_method == 'weighted_average':
                    if (len(stage_predictions) > 1 and stage_idx < len(self.stage_performances_) and len(self.stage_performances_[stage_idx]) == len(stage_predictions)):
                        weights = np.array(self.stage_performances_[stage_idx])
                        weights = weights / weights.sum()
                        if all(p.shape[1] == stage_predictions[0].shape[1] for p in stage_predictions):
                            stage_features = np.average(stage_predictions, axis=0, weights=weights)
                            if self.passthrough_original and X_original is not None:
                                X_current = np.hstack([X_original, stage_features])
                            else:
                                X_current = stage_features
                        else:
                            X_current = np.hstack(stage_predictions)
                    else:
                        X_current = np.hstack(stage_predictions)
            probabilities = []
            for estimator in self.stages_[-1]:
                if hasattr(estimator, 'predict_proba'):
                    probabilities.append(estimator.predict_proba(X_current))
            if probabilities:
                return np.mean(probabilities, axis=0)
            else:
                n_classes = len(self.classes_)
                predictions = self.predict(X)
                proba = np.zeros((len(predictions), n_classes))
                for i, pred in enumerate(predictions):
                    proba[i, pred] = 1.0
                return proba

    def transform(self, X):
        if not hasattr(self, 'stages_') or not self.stages_:
            raise ValueError("Model must be fitted before transform can be called")
        X_current = X.copy()
        X_original = X.copy() if self.passthrough_original else None
        for stage_idx, stage_models in enumerate(self.stages_):
            stage_predictions = []
            for estimator in stage_models:
                if hasattr(estimator, 'predict_proba'):
                    pred_proba = estimator.predict_proba(X_current)
                    if pred_proba.shape[1] > 2:
                        stage_predictions.append(pred_proba)
                    else:
                        stage_predictions.append(pred_proba[:, 1:2])
                else:
                    pred = estimator.predict(X_current)
                    stage_predictions.append(pred.reshape(-1, 1))
            if stage_idx == len(self.stages_) - 1:
                final_features = np.hstack(stage_predictions)
                if self.passthrough_original and X_original is not None:
                    final_features = np.hstack([X_original, final_features])
                return final_features
            stage_features = np.hstack(stage_predictions)
            if self.combine_method == 'concatenate':
                if self.passthrough_original and X_original is not None:
                    X_current = np.hstack([X_original, stage_features])
                else:
                    X_current = stage_features
            elif self.combine_method == 'average':
                if len(stage_predictions) > 1 and all(p.shape[1] == stage_predictions[0].shape[1] for p in stage_predictions):
                    stage_features = np.mean(stage_predictions, axis=0)
                    if self.passthrough_original and X_original is not None:
                        X_current = np.hstack([X_original, stage_features])
                    else:
                        X_current = stage_features
                else:
                    X_current = np.hstack(stage_predictions)
            elif self.combine_method == 'weighted_average':
                if (len(stage_predictions) > 1 and stage_idx < len(self.stage_performances_) and len(self.stage_performances_[stage_idx]) == len(stage_predictions)):
                    weights = np.array(self.stage_performances_[stage_idx])
                    weights = weights / weights.sum()
                    if all(p.shape[1] == stage_predictions[0].shape[1] for p in stage_predictions):
                        stage_features = np.average(stage_predictions, axis=0, weights=weights)
                        if self.passthrough_original and X_original is not None:
                            X_current = np.hstack([X_original, stage_features])
                        else:
                            X_current = stage_features
                    else:
                        X_current = np.hstack(stage_predictions)
                else:
                    X_current = np.hstack(stage_predictions)
        return X_current

class MultiStageStackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, stages_config: List[Dict], final_estimator=None, cv=5, n_jobs=None, passthrough=False):
        self.stages_config = stages_config
        self.final_estimator = final_estimator
        self.cv = cv
        self.n_jobs = n_jobs
        self.passthrough = passthrough
        self.stages_ = []
        self.final_stage_ = None

    def _create_estimator_from_config(self, estimator_config: Dict, available_models: Dict):
        model_name = estimator_config['model']
        model_params = estimator_config.get('params', {})
        model_class = available_models[model_name]['class']
        return model_class(**model_params)

    def fit(self, X, y, available_models):
        self.classes_ = np.unique(y)
        X_current = X.copy()
        for stage_idx, stage_config in enumerate(self.stages_config):
            logger.info(f"Fitting stage {stage_idx + 1}/{len(self.stages_config)}")
            stage_estimators = []
            stage_predictions = []
            for estimator_config in stage_config['estimators']:
                estimator = self._create_estimator_from_config(estimator_config, available_models)
                estimator.fit(X_current, y)
                stage_estimators.append(estimator)
                if hasattr(estimator, 'predict_proba'):
                    cv_pred = cross_val_predict(estimator, X_current, y, cv=self.cv, method='predict_proba')
                    if cv_pred.shape[1] > 2:
                        stage_predictions.append(cv_pred)
                    else:
                        stage_predictions.append(cv_pred[:, 1:2])
                else:
                    cv_pred = cross_val_predict(estimator, X_current, y, cv=self.cv)
                    stage_predictions.append(cv_pred.reshape(-1, 1))
            self.stages_.append(stage_estimators)
            if stage_idx < len(self.stages_config) - 1:
                stage_features = np.hstack(stage_predictions)
                if stage_config.get('include_original', False):
                    X_current = np.hstack([X_current, stage_features])
                else:
                    X_current = stage_features
                logger.info(f"Stage {stage_idx + 1} output shape: {X_current.shape}")
        if self.final_estimator is not None:
            final_stage_predictions = []
            for estimator in self.stages_[-1]:
                if hasattr(estimator, 'predict_proba'):
                    cv_pred = cross_val_predict(estimator, X_current, y, cv=self.cv, method='predict_proba')
                    if cv_pred.shape[1] > 2:
                        final_stage_predictions.append(cv_pred)
                    else:
                        final_stage_predictions.append(cv_pred[:, 1:2])
                else:
                    cv_pred = cross_val_predict(estimator, X_current, y, cv=self.cv)
                    final_stage_predictions.append(cv_pred.reshape(-1, 1))
            final_features = np.hstack(final_stage_predictions)
            if self.passthrough:
                final_features = np.hstack([X, final_features])
            self.final_stage_ = clone(self.final_estimator)
            self.final_stage_.fit(final_features, y)
            logger.info(f"Final estimator fitted with input shape: {final_features.shape}")
        return self

    def predict(self, X):
        X_current = X.copy()
        for stage_idx, stage_estimators in enumerate(self.stages_[:-1]):
            stage_predictions = []
            for estimator in stage_estimators:
                if hasattr(estimator, 'predict_proba'):
                    pred_proba = estimator.predict_proba(X_current)
                    if pred_proba.shape[1] > 2:
                        stage_predictions.append(pred_proba)
                    else:
                        stage_predictions.append(pred_proba[:, 1:2])
                else:
                    pred = estimator.predict(X_current)
                    stage_predictions.append(pred.reshape(-1, 1))
            stage_features = np.hstack(stage_predictions)
            if self.stages_config[stage_idx].get('include_original', False):
                X_current = np.hstack([X_current, stage_features])
            else:
                X_current = stage_features
        if self.final_stage_ is not None:
            final_stage_predictions = []
            for estimator in self.stages_[-1]:
                if hasattr(estimator, 'predict_proba'):
                    pred_proba = estimator.predict_proba(X_current)
                    if pred_proba.shape[1] > 2:
                        final_stage_predictions.append(pred_proba)
                    else:
                        final_stage_predictions.append(pred_proba[:, 1:2])
                else:
                    pred = estimator.predict(X_current)
                    final_stage_predictions.append(pred.reshape(-1, 1))
            final_features = np.hstack(final_stage_predictions)
            if self.passthrough:
                final_features = np.hstack([X, final_features])
            return self.final_stage_.predict(final_features)
        else:
            predictions = []
            for estimator in self.stages_[-1]:
                predictions.append(estimator.predict(X_current))
            predictions = np.array(predictions).T
            return np.array([np.bincount(row).argmax() for row in predictions])

    def predict_proba(self, X):
        if self.final_stage_ is not None and hasattr(self.final_stage_, 'predict_proba'):
            X_current = X.copy()
            for stage_idx, stage_estimators in enumerate(self.stages_[:-1]):
                stage_predictions = []
                for estimator in stage_estimators:
                    if hasattr(estimator, 'predict_proba'):
                        pred_proba = estimator.predict_proba(X_current)
                        if pred_proba.shape[1] > 2:
                            stage_predictions.append(pred_proba)
                        else:
                            stage_predictions.append(pred_proba[:, 1:2])
                    else:
                        pred = estimator.predict(X_current)
                        stage_predictions.append(pred.reshape(-1, 1))
                stage_features = np.hstack(stage_predictions)
                if self.stages_config[stage_idx].get('include_original', False):
                    X_current = np.hstack([X_current, stage_features])
                else:
                    X_current = stage_features
            final_stage_predictions = []
            for estimator in self.stages_[-1]:
                if hasattr(estimator, 'predict_proba'):
                    pred_proba = estimator.predict_proba(X_current)
                    if pred_proba.shape[1] > 2:
                        final_stage_predictions.append(pred_proba)
                    else:
                        final_stage_predictions.append(pred_proba[:, 1:2])
                else:
                    pred = estimator.predict(X_current)
                    final_stage_predictions.append(pred.reshape(-1, 1))
            final_features = np.hstack(final_stage_predictions)
            if self.passthrough:
                final_features = np.hstack([X, final_features])
            return self.final_stage_.predict_proba(final_features)
        else:
            X_current = X.copy()
            for stage_idx, stage_estimators in enumerate(self.stages_[:-1]):
                stage_predictions = []
                for estimator in stage_estimators:
                    if hasattr(estimator, 'predict_proba'):
                        pred_proba = estimator.predict_proba(X_current)
                        if pred_proba.shape[1] > 2:
                            stage_predictions.append(pred_proba)
                        else:
                            stage_predictions.append(pred_proba[:, 1:2])
                    else:
                        pred = estimator.predict(X_current)
                        stage_predictions.append(pred.reshape(-1, 1))
                stage_features = np.hstack(stage_predictions)
                if self.stages_config[stage_idx].get('include_original', False):
                    X_current = np.hstack([X_current, stage_features])
                else:
                    X_current = stage_features
            probabilities = []
            for estimator in self.stages_[-1]:
                if hasattr(estimator, 'predict_proba'):
                    probabilities.append(estimator.predict_proba(X_current))
            if probabilities:
                return np.mean(probabilities, axis=0)
            else:
                n_classes = len(self.classes_)
                predictions = self.predict(X)
                proba = np.zeros((len(predictions), n_classes))
                for i, pred in enumerate(predictions):
                    proba[i, pred] = 1.0
                return proba

class AdvancedEnsembleBuilder:
    def __init__(self, available_models: Dict):
        self.available_models = available_models
        self.parameter_validator = ModelParameterValidator()

    def _create_model_instance(self, model_config: Dict, y_train: pd.Series = None) -> Any:
        model_name = model_config['model']
        model_params = model_config.get('params', {})
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not available")
        model_info = self.available_models[model_name]
        model_class_name = model_info['class'].__name__
        validated_params = self.parameter_validator.validate_and_convert_parameters(model_class_name, model_params)
        if y_train is not None:
            n_classes = len(y_train.unique())
            if model_name == "XGBoost" and XGBOOST_AVAILABLE:
                if n_classes > 2:
                    validated_params['objective'] = 'multi:softprob'
                    validated_params['num_class'] = n_classes
            elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
                if n_classes > 2:
                    validated_params['objective'] = 'multiclass'
                    validated_params['num_class'] = n_classes
            elif model_name == "CatBoost" and CATBOOST_AVAILABLE:
                if n_classes > 2:
                    validated_params['loss_function'] = 'MultiClass'
        return model_info['class'](**validated_params)

    def create_voting_ensemble(self, models_config: List[Dict], voting: str = 'soft', y_train: pd.Series = None) -> VotingClassifier:
        estimators = []
        for i, config in enumerate(models_config):
            model_instance = self._create_model_instance(config, y_train)
            estimators.append((f"{config['model']}_{i}", model_instance))
        return VotingClassifier(estimators=estimators, voting=voting)

    def create_simple_stacking_ensemble(self, base_models_config: List[Dict], meta_model_config: Dict, cv: int = 5, y_train: pd.Series = None) -> StackingClassifier:
        estimators = []
        for i, config in enumerate(base_models_config):
            model_instance = self._create_model_instance(config, y_train)
            estimators.append((f"{config['model']}_{i}", model_instance))
        meta_model = self._create_model_instance(meta_model_config, y_train)
        return StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=cv, stack_method='auto')

    def create_multi_stage_stacking_ensemble(self, stages_config: List[Dict], final_estimator_config: Dict = None, cv: int = 5, passthrough: bool = False, y_train: pd.Series = None) -> MultiStageStackingClassifier:
        final_estimator = None
        if final_estimator_config:
            final_estimator = self._create_model_instance(final_estimator_config, y_train)
        return MultiStageStackingClassifier(stages_config=stages_config, final_estimator=final_estimator, cv=cv, passthrough=passthrough)

    def create_staged_ensemble(self, architecture: List[Dict], final_meta_model_config: Dict = None, cv: int = 5, stage_method: str = 'stacking', combine_method: str = 'concatenate', passthrough_original: bool = False, y_train: pd.Series = None) -> StagedEnsembleClassifier:
        final_meta_model = None
        if final_meta_model_config:
            final_meta_model = self._create_model_instance(final_meta_model_config, y_train)
        return StagedEnsembleClassifier(architecture=architecture, final_meta_model=final_meta_model, cv=cv, stage_method=stage_method, combine_method=combine_method, passthrough_original=passthrough_original)

    def create_bagging_ensemble(self, model_config: Dict, n_estimators: int = 10, y_train: pd.Series = None) -> BaggingClassifier:
        base_estimator = self._create_model_instance(model_config, y_train)
        return BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators, random_state=42)

    def create_hybrid_ensemble(self, ensemble_config: Dict, y_train: pd.Series = None):
        ensemble_type = ensemble_config.get('type', 'hybrid')
        if ensemble_type == 'voting_stacking':
            voting_configs = ensemble_config.get('voting_ensembles', [])
            stacking_config = ensemble_config.get('stacking_config', {})
            base_ensembles = []
            for i, voting_config in enumerate(voting_configs):
                voting_ensemble = self.create_voting_ensemble(voting_config['models'], voting_config.get('voting', 'soft'), y_train)
                base_ensembles.append((f"voting_ensemble_{i}", voting_ensemble))
            meta_model = self._create_model_instance(stacking_config['meta_model'], y_train)
            return StackingClassifier(estimators=base_ensembles, final_estimator=meta_model, cv=stacking_config.get('cv', 5))
        elif ensemble_type == 'bagging_stacking':
            bagging_configs = ensemble_config.get('bagging_ensembles', [])
            stacking_config = ensemble_config.get('stacking_config', {})
            base_ensembles = []
            for i, bagging_config in enumerate(bagging_configs):
                bagging_ensemble = self.create_bagging_ensemble(bagging_config['base_model'], bagging_config.get('n_estimators', 10), y_train)
                base_ensembles.append((f"bagging_ensemble_{i}", bagging_ensemble))
            meta_model = self._create_model_instance(stacking_config['meta_model'], y_train)
            return StackingClassifier(estimators=base_ensembles, final_estimator=meta_model, cv=stacking_config.get('cv', 5))
        else:
            raise ValueError(f"Unknown hybrid ensemble type: {ensemble_type}")

class EnhancedModelBuilder:
    def __init__(self, intel_path: str = "intel.yaml"):
        logger.info(f"Initializing Enhanced ModelBuilder with intel file: {intel_path}")
        self.intel_path = intel_path
        self.intel = self._load_intel()
        self.dataset_name = self.intel.get('dataset_name')
        self.target_column = self.intel.get('target_column')
        self.train_data_path = self.intel.get('train_selected_path', self.intel.get('train_transformed_path'))
        self.test_data_path = self.intel.get('test_selected_path', self.intel.get('test_transformed_path'))
        self.model_dir = Path(f"model/model_{self.dataset_name}")
        self.model_path = self.model_dir / "model.pkl"
        self.parameter_validator = ModelParameterValidator()
        self.available_models = self._get_available_models_corrected()
        self.ensemble_builder = AdvancedEnsembleBuilder(self.available_models)
        logger.info(f"Enhanced ModelBuilder initialized for dataset: {self.dataset_name}")

    def _load_intel(self) -> Dict[str, Any]:
        try:
            with open(self.intel_path, 'r') as file:
                intel = yaml.safe_load(file)
            logger.info(f"Successfully loaded intel from {self.intel_path}")
            return intel
        except Exception as e:
            logger.error(f"Failed to load intel file: {e}")
            raise

    def _get_available_models_corrected(self) -> Dict[str, Dict[str, Any]]:
        models = {
            "Logistic Regression": {"class": LogisticRegression, "params": {"C": 1.0, "max_iter": 1000, "solver": "liblinear", "random_state": 42, "n_jobs": -1}},
            "Ridge Classifier": {"class": RidgeClassifier, "params": {"alpha": 1.0, "fit_intercept": True, "max_iter": 1000}},
            "SGD Classifier": {"class": SGDClassifier, "params": {"loss": "hinge", "penalty": "l2", "alpha": 0.0001, "max_iter": 1000, "random_state": 42}},
            "Passive Aggressive": {"class": PassiveAggressiveClassifier, "params": {"C": 1.0, "max_iter": 1000, "tol": 1e-3, "random_state": 42}},
            "Decision Tree": {"class": DecisionTreeClassifier, "params": {"max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1, "criterion": "gini", "random_state": 42}},
            "Random Forest": {"class": RandomForestClassifier, "params": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "n_jobs": -1, "random_state": 42}},
            "Gradient Boosting": {"class": GradientBoostingClassifier, "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 1.0, "random_state": 42}},
            "AdaBoost": {"class": AdaBoostClassifier, "params": {"n_estimators": 50, "learning_rate": 1.0, "algorithm": "SAMME", "random_state": 42}},
            "Extra Trees": {"class": ExtraTreesClassifier, "params": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "n_jobs": -1, "random_state": 42}},
            "K-Nearest Neighbors": {"class": KNeighborsClassifier, "params": {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto", "n_jobs": -1}},
            "Support Vector Classifier": {"class": SVC, "params": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": True, "random_state": 42}},
            "MLP Classifier": {"class": MLPClassifier, "params": {"hidden_layer_sizes": (100,), "activation": "relu", "solver": "adam", "max_iter": 200, "random_state": 42}},
            "Gaussian Naive Bayes": {"class": GaussianNB, "params": {"var_smoothing": 1e-9}},
            "Multinomial Naive Bayes": {"class": MultinomialNB, "params": {"alpha": 1.0, "fit_prior": True}},
            "Bernoulli Naive Bayes": {"class": BernoulliNB, "params": {"alpha": 1.0, "binarize": 0.0, "fit_prior": True}},
            "Linear Discriminant Analysis": {"class": LinearDiscriminantAnalysis, "params": {"solver": "svd", "shrinkage": None}},
            "Quadratic Discriminant Analysis": {"class": QuadraticDiscriminantAnalysis, "params": {"reg_param": 0.0}},
        }
        try:
            import xgboost as xgb
            models["XGBoost"] = {"class": xgb.XGBClassifier, "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 0.8, "colsample_bytree": 0.8, "objective": "binary:logistic", "n_jobs": -1, "eval_metric": "logloss", "random_state": 42, "verbosity": 0}}
        except ImportError:
            pass
        try:
            import lightgbm as lgb
            models["LightGBM"] = {"class": lgb.LGBMClassifier, "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": -1, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8, "objective": "binary", "n_jobs": -1, "random_state": 42, "verbose": -1}}
        except ImportError:
            pass
        try:
            import catboost as cb
            models["CatBoost"] = {"class": cb.CatBoostClassifier, "params": {"iterations": 100, "learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 3.0, "loss_function": "Logloss", "verbose": False, "random_state": 42}}
        except ImportError:
            pass
        return models

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        logger.info("Loading Data")
        try:
            train_data = pd.read_csv(self.train_data_path)
            logger.info(f"Loaded training data from {self.train_data_path}")
            logger.info(f"Training data shape: {train_data.shape}")
            test_data = pd.read_csv(self.test_data_path)
            logger.info(f"Loaded test data from {self.test_data_path}")
            logger.info(f"Test data shape: {test_data.shape}")
            X_train = train_data.drop(columns=[self.target_column])
            y_train = train_data[self.target_column]
            X_test = test_data.drop(columns=[self.target_column])
            y_test = test_data[self.target_column]
            logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def get_available_models(self) -> Dict[str, Dict]:
        models_info = {}
        for model_name, model_data in self.available_models.items():
            models_info[model_name] = {"params": model_data["params"]}
        return models_info

    def build_staged_ensemble(self, staged_config: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Building staged ensemble model with hierarchical architecture")
        try:
            X_train, y_train, X_test, y_test = self.load_data()
            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Target classes: {sorted(y_train.unique())}")
            architecture = staged_config['architecture']
            final_meta_model_config = staged_config.get('final_meta_model')
            cv = staged_config.get('cv', 5)
            stage_method = staged_config.get('stage_method', 'stacking')
            combine_method = staged_config.get('combine_method', 'concatenate')
            passthrough_original = staged_config.get('passthrough_original', False)
            logger.info(f"Staged ensemble architecture: {len(architecture)} stages")
            for i, stage in enumerate(architecture):
                stage_model_count = len(stage.get('models', []))
                logger.info(f"Stage {i + 1}: {stage_model_count} models")
            model = self.ensemble_builder.create_staged_ensemble(architecture=architecture, final_meta_model_config=final_meta_model_config, cv=cv, stage_method=stage_method, combine_method=combine_method, passthrough_original=passthrough_original, y_train=y_train)
            logger.info("Training staged ensemble model...")
            start_time = time.time()
            model.fit(X_train, y_train, self.available_models)
            training_time = time.time() - start_time
            logger.info(f"Staged ensemble training completed in {training_time:.2f} seconds")
            model_path = self.save_model(model, "staged_ensemble")
            logger.info("Staged ensemble model training completed successfully")
            return {'model_name': 'staged_ensemble', 'model_path': model_path, 'staged_config': staged_config, 'training_time': training_time, 'architecture_summary': {'total_stages': len(architecture), 'stage_sizes': [len(stage.get('models', [])) for stage in architecture], 'combine_method': combine_method, 'stage_method': stage_method}, 'status': 'success'}
        except Exception as e:
            logger.error(f"Error building staged ensemble model: {str(e)}")
            logger.error(f"Staged config: {staged_config}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def build_ensemble_model(self, ensemble_config: Dict[str, Any]) -> Dict[str, Any]:
        ensemble_type = ensemble_config.get('type')
        logger.info(f"Building ensemble model: {ensemble_type}")
        try:
            X_train, y_train, X_test, y_test = self.load_data()
            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Target classes: {sorted(y_train.unique())}")
            if ensemble_type == 'staged_ensemble':
                return self.build_staged_ensemble(ensemble_config)
            elif ensemble_type == 'voting':
                logger.info("Creating voting classifier ensemble")
                model = self.ensemble_builder.create_voting_ensemble(ensemble_config['models'], ensemble_config.get('voting', 'soft'), y_train)
            elif ensemble_type == 'stacking':
                logger.info("Creating stacking classifier ensemble")
                model = self.ensemble_builder.create_simple_stacking_ensemble(ensemble_config['base_models'], ensemble_config['meta_model'], ensemble_config.get('cv', 5), y_train)
            elif ensemble_type == 'multi_stage_stacking':
                logger.info("Creating multi-stage stacking classifier ensemble")
                model = self.ensemble_builder.create_multi_stage_stacking_ensemble(ensemble_config['stages'], ensemble_config.get('final_estimator'), ensemble_config.get('cv', 5), ensemble_config.get('passthrough', False), y_train)
            elif ensemble_type == 'bagging':
                logger.info("Creating bagging classifier ensemble")
                model = self.ensemble_builder.create_bagging_ensemble(ensemble_config['base_model'], ensemble_config.get('n_estimators', 10), y_train)
            elif ensemble_type in ['voting_stacking', 'bagging_stacking']:
                logger.info(f"Creating hybrid ensemble: {ensemble_type}")
                model = self.ensemble_builder.create_hybrid_ensemble(ensemble_config, y_train)
            else:
                raise ValueError(f"Unknown ensemble type: {ensemble_type}")
            logger.info("Training ensemble model...")
            start_time = time.time()
            if isinstance(model, (MultiStageStackingClassifier, StagedEnsembleClassifier)):
                model.fit(X_train, y_train, self.available_models)
            else:
                model.fit(X_train, y_train)
            training_time = time.time() - start_time
            logger.info(f"Ensemble training completed in {training_time:.2f} seconds")
            model_path = self.save_model(model, f"ensemble_{ensemble_type}")
            logger.info("Ensemble model training completed successfully")
            return {'model_name': f"ensemble_{ensemble_type}", 'model_path': model_path, 'ensemble_config': ensemble_config, 'training_time': training_time, 'status': 'success'}
        except Exception as e:
            logger.error(f"Error building ensemble model: {str(e)}")
            logger.error(f"Ensemble config: {ensemble_config}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def build_single_model(self, model_name: str, custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info(f"Building single model: {model_name}")
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not available")
        model_info = self.available_models[model_name].copy()
        if custom_params:
            validated_custom_params = self.parameter_validator.validate_and_convert_parameters(model_info['class'].__name__, custom_params)
            model_info['params'].update(validated_custom_params)
        try:
            X_train, y_train, X_test, y_test = self.load_data()
            model = self._create_model_instance(model_name, model_info, y_train)
            logger.info(f"Training {model_name}...")
            model.fit(X_train, y_train)
            model_path = self.save_model(model, model_name)
            return {'model_name': model_name, 'model_path': model_path, 'parameters': model_info['params'], 'status': 'success'}
        except Exception as e:
            logger.error(f"Error building single model: {str(e)}")
            raise

    def build_parallel_models(self, models_config: List[Dict[str, Any]], max_workers: int = 4) -> Dict[str, Any]:
        logger.info(f"Building {len(models_config)} models in parallel")
        try:
            X_train, y_train, X_test, y_test = self.load_data()
            models_dict = {}
            for i, config in enumerate(models_config):
                model_name = config['model_name']
                custom_params = config.get('custom_params', {})
                if model_name not in self.available_models:
                    raise ValueError(f"Model '{model_name}' not available")
                model_info = self.available_models[model_name].copy()
                if custom_params:
                    validated_custom_params = self.parameter_validator.validate_and_convert_parameters(model_info['class'].__name__, custom_params)
                    model_info['params'].update(validated_custom_params)
                model = self._create_model_instance(model_name, model_info, y_train)
                models_dict[f"{model_name}_{i}"] = model
            trainer = ParallelModelTrainer(X_train, y_train, X_test, y_test)
            results = trainer.train_models_parallel(models_dict, max_workers)
            cleaned_results = {}
            for model_id, result in results.items():
                cleaned_result = result.copy()
                cleaned_result.pop('model', None)
                cleaned_results[model_id] = cleaned_result
            logger.info("Parallel model training completed")
            return {'status': 'success', 'results': cleaned_results, 'total_models': len(models_config)}
        except Exception as e:
            logger.error(f"Error in parallel model building: {str(e)}")
            raise

    def _create_model_instance(self, model_name: str, model_info: Dict[str, Any], y_train: pd.Series) -> Any:
        validated_params = self.parameter_validator.validate_and_convert_parameters(model_info['class'].__name__, model_info['params'].copy())
        if model_name == "XGBoost" and XGBOOST_AVAILABLE:
            n_classes = len(y_train.unique())
            if n_classes > 2:
                validated_params['objective'] = 'multi:softprob'
                validated_params['num_class'] = n_classes
            logger.info(f"XGBoost configured for {n_classes} classes")
        elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
            n_classes = len(y_train.unique())
            if n_classes > 2:
                validated_params['objective'] = 'multiclass'
                validated_params['num_class'] = n_classes
            logger.info(f"LightGBM configured for {n_classes} classes")
        elif model_name == "CatBoost" and CATBOOST_AVAILABLE:
            n_classes = len(y_train.unique())
            if n_classes > 2:
                validated_params['loss_function'] = 'MultiClass'
            logger.info(f"CatBoost configured for {n_classes} classes")
        try:
            model_instance = model_info['class'](**validated_params)
            logger.info(f"Created {model_name} instance with validated parameters")
            return model_instance
        except Exception as e:
            logger.error(f"Error creating {model_name} instance: {e}")
            logger.error(f"Parameters: {validated_params}")
            raise

    def save_model(self, model: Any, model_name: str) -> str:
        logger.info("Saving Model")
        os.makedirs(self.model_dir, exist_ok=True)
        try:
            with open(self.model_path, 'wb') as f:
                cloudpickle.dump(model, f)
            logger.info(f"Model saved to {self.model_path}")
            model_path_str = str(self.model_path)
            self.intel['model_path'] = model_path_str
            self.intel['model_name'] = model_name
            self.intel['model_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.intel_path, 'w') as f:
                yaml.dump(self.intel, f, default_flow_style=False)
            logger.info(f"Updated intel.yaml with model path: {model_path_str}")
            return model_path_str
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def select_best_model(self, parallel_results: Dict[str, Any], selection_metric: str = 'accuracy') -> Dict[str, Any]:
        logger.info(f"Selecting best model based on {selection_metric}")
        best_model_id = None
        best_score = -float('inf') if selection_metric != 'log_loss' else float('inf')
        for model_id, result in parallel_results['results'].items():
            if result['status'] == 'success' and selection_metric in result['metrics']:
                score = result['metrics'][selection_metric]
                if selection_metric == 'log_loss':
                    if score < best_score:
                        best_score = score
                        best_model_id = model_id
                else:
                    if score > best_score:
                        best_score = score
                        best_model_id = model_id
        if best_model_id:
            best_model_path = parallel_results['results'][best_model_id]['model_path']
            with open(best_model_path, 'rb') as f:
                best_model = cloudpickle.load(f)
            model_path = self.save_model(best_model, best_model_id)
            logger.info(f"Best model selected: {best_model_id} with {selection_metric}: {best_score:.4f}")
            return {'best_model_id': best_model_id, 'best_score': best_score, 'model_path': model_path, 'metrics': parallel_results['results'][best_model_id]['metrics'], 'status': 'success'}
        else:
            raise ValueError("No valid models found for selection")

class ParallelModelTrainer:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.results = {}
        self.lock = threading.Lock()

    def train_and_evaluate_model(self, model_id: str, model: Any) -> Dict[str, Any]:
        try:
            logger.info(f"Training model {model_id}")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(self.X_test)
                except Exception as e:
                    logger.warning(f"Model {model_id} does not support predict_proba: {str(e)}")
            metrics = {'accuracy': accuracy_score(self.y_test, y_pred), 'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0), 'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0), 'f1_score': f1_score(self.y_test, y_pred, average='weighted', zero_division=0)}
            if y_pred_proba is not None:
                try:
                    n_classes = len(np.unique(self.y_test))
                    if n_classes == 2:
                        metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                    else:
                        metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='weighted')
                except Exception as e:
                    logger.warning(f"Failed to calculate ROC-AUC for model {model_id}: {str(e)}")
            model_path = f"model/tmp/{model_id}.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                cloudpickle.dump(model, f)
            result = {'model_id': model_id, 'model_path': model_path, 'metrics': metrics, 'status': 'success', 'training_time': datetime.now().isoformat()}
            logger.info(f"Model {model_id} training completed - Accuracy: {metrics['accuracy']:.4f}")
            return result
        except Exception as e:
            logger.error(f"Error training model {model_id}: {str(e)}")
            return {'model_id': model_id, 'model': None, 'metrics': {}, 'status': 'failed', 'error': str(e)}

    def train_models_parallel(self, models_dict: Dict[str, Any], max_workers: int = 4) -> Dict[str, Any]:
        logger.info(f"Starting parallel training of {len(models_dict)} models")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {executor.submit(self.train_and_evaluate_model, model_id, model): model_id for model_id, model in models_dict.items()}
            results = {}
            for future in as_completed(future_to_model):
                result = future.result()
                results[result['model_id']] = result
        logger.info("Parallel training completed")
        return results

def build_single_model_api_patched(model_name: str, custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    builder = EnhancedModelBuilder()
    return builder.build_single_model(model_name, custom_params)

def build_parallel_models_api_patched(models_config: List[Dict[str, Any]], max_workers: int = 4) -> Dict[str, Any]:
    builder = EnhancedModelBuilder()
    return builder.build_parallel_models(models_config, max_workers)

def build_ensemble_model_api(ensemble_config: Dict[str, Any]) -> Dict[str, Any]:
    builder = EnhancedModelBuilder()
    try:
        if 'type' not in ensemble_config:
            raise ValueError("Ensemble config must include 'type' field")
        ensemble_type = ensemble_config['type']
        logger.info(f"Building ensemble model of type: {ensemble_type}")
        if ensemble_type == 'voting':
            if 'models' not in ensemble_config:
                raise ValueError("Voting ensemble requires 'models' field")
            if len(ensemble_config['models']) < 2:
                raise ValueError("Voting ensemble requires at least 2 models")
        elif ensemble_type == 'stacking':
            if 'base_models' not in ensemble_config:
                raise ValueError("Stacking ensemble requires 'base_models' field")
            if 'meta_model' not in ensemble_config:
                raise ValueError("Stacking ensemble requires 'meta_model' field")
            if len(ensemble_config['base_models']) < 2:
                raise ValueError("Stacking ensemble requires at least 2 base models")
        elif ensemble_type == 'bagging':
            if 'base_model' not in ensemble_config:
                raise ValueError("Bagging ensemble requires 'base_model' field")
            if 'n_estimators' not in ensemble_config:
                ensemble_config['n_estimators'] = 10
        elif ensemble_type == 'staged_ensemble':
            if 'architecture' not in ensemble_config:
                raise ValueError("Staged ensemble requires 'architecture' field")
            if not isinstance(ensemble_config['architecture'], list) or len(ensemble_config['architecture']) < 2:
                raise ValueError("Staged ensemble requires at least 2 stages in architecture")
        logger.info(f"Ensemble configuration: {ensemble_config}")
        return builder.build_ensemble_model(ensemble_config)
    except Exception as e:
        logger.error(f"Error in build_ensemble_model_api: {str(e)}")
        logger.error(f"Ensemble config was: {ensemble_config}")
        raise

def build_staged_ensemble_api(staged_config: Dict[str, Any]) -> Dict[str, Any]:
    builder = EnhancedModelBuilder()
    try:
        if 'architecture' not in staged_config:
            raise ValueError("Staged ensemble config must include 'architecture' field")
        architecture = staged_config['architecture']
        if not isinstance(architecture, list) or len(architecture) < 2:
            raise ValueError("Staged ensemble requires at least 2 stages in architecture")
        for i, stage in enumerate(architecture):
            if 'models' not in stage:
                raise ValueError(f"Stage {i + 1} must include 'models' field")
            if not isinstance(stage['models'], list) or len(stage['models']) == 0:
                raise ValueError(f"Stage {i + 1} must have at least 1 model")
        logger.info(f"Building staged ensemble with {len(architecture)} stages")
        return builder.build_staged_ensemble(staged_config)
    except Exception as e:
        logger.error(f"Error in build_staged_ensemble_api: {str(e)}")
        logger.error(f"Staged config was: {staged_config}")
        raise

def select_best_model_api_patched(parallel_results: Dict[str, Any], selection_metric: str = 'accuracy') -> Dict[str, Any]:
    builder = EnhancedModelBuilder()
    return builder.select_best_model(parallel_results, selection_metric)

if __name__ == "__main__":
    print("Enhanced Model Building Script with Complex Ensemble Support and Staged Architecture")