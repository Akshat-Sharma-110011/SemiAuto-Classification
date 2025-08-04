import os
import yaml
import pandas as pd
import numpy as np
import cloudpickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings('ignore')

# Import logger
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from semiauto_classification.logger import get_logger, section

# Meta-heuristic algorithms
try:
    from geneticalgorithm import geneticalgorithm as ga
except ImportError:
    ga = None

try:
    from mealpy.human_based import (
        BRO, BSO, CA, CHIO, FBIO, GSKA, HBO, HCO,
        ICA, LCO, QSA, SARO, SPBO, SSDO, TLO
    )
    from mealpy.physics_based import (
        ASO, ArchOA, CDO, EFO, EO, EVO, FLA,
        HGSO, MVO, NRO, RIME, SA, TWO, WDO
    )
    from mealpy.math_based import (
        AOA, CEM, CGO, CircleSA, GBO, HC, INFO, PSS,
        RUN, SCA, SHIO, TS
    )
    from mealpy.evolutionary_based import (
        EP,  # BaseEP, LevyEP
        ES,  # BaseES, LevyES
        MA,  # Memetic Algorithm
        GA,  # Genetic Algorithm
        DE  # Differential Evolution and variants (JADE, SADE, SHADE…)
    )
    from mealpy.swarm_based import (
        ABC, ACOR, AGTO, ALO, AO, ARO, AVOA, BA, BES, BFO, BSA, BeesA,
        COA, CSA, CSO, CoatiOA, DMOA, DO, EHO, ESOA, FA, FFA, FFO, FOA,
        FOX, GJO, GOA, GTO, GWO, HBA, HGS, HHO, JA, MFO, MGO, MPA, MRFO,
        MSA, NGO, NMRA, OOA, PFA, POA, PSO, SCSO, SFO, SHO, SLO, SRSR,
        SSA, SSO, SSpiderA, SSpiderO, STO, SeaHO, ServalOA, TDO, TSO,
        WOA, WaOA, ZOA,
    )

    MEALPY_AVAILABLE = True
except ImportError:
    MEALPY_AVAILABLE = False

try:
    from pyswarm import pso as pyswarm_pso

    PYSWARM_AVAILABLE = True
except ImportError:
    PYSWARM_AVAILABLE = False

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class MetaHeuristicFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Meta-heuristic based feature selector using various optimization algorithms.
    """

    def __init__(self, algorithm='genetic_algorithm', n_features=None,
                 max_iter=50, population_size=20, random_state=42):
        """
        Initialize the meta-heuristic feature selector.

        Parameters:
        -----------
        algorithm : str
            The meta-heuristic algorithm to use
        n_features : int or None
            Number of features to select. If None, will be determined automatically
        max_iter : int
            Maximum number of iterations
        population_size : int
            Population size for the algorithm
        random_state : int
            Random state for reproducibility
        """
        self.algorithm = algorithm
        self.n_features = n_features
        self.max_iter = max_iter
        self.population_size = population_size
        self.random_state = random_state
        self.selected_features_ = None
        self.feature_names_ = None
        self.logger = get_logger(__name__)

        # Available algorithms
        self.available_algorithms = {
            'genetic_algorithm': self._genetic_algorithm,
            'particle_swarm': self._particle_swarm,
            'ant_colony': self._ant_colony,
            'simulated_annealing': self._simulated_annealing,
            'whale_optimization': self._whale_optimization,
            'grey_wolf': self._grey_wolf_optimization,
            'differential_evolution': self._differential_evolution,
            'artificial_bee_colony': self._artificial_bee_colony,
            'slime_mould': self._slime_mould_algorithm,
            'sine_cosine': self._sine_cosine_algorithm,
            'optuna_bayesian': self._optuna_bayesian,
            'random_search': self._random_search
        }

    def get_available_algorithms(self) -> List[str]:
        """Get list of available algorithms based on installed packages."""
        available = ['random_search']  # Always available

        if ga is not None:
            available.append('genetic_algorithm')

        if MEALPY_AVAILABLE:
            available.extend([
                'particle_swarm', 'ant_colony', 'simulated_annealing',
                'whale_optimization', 'grey_wolf', 'differential_evolution',
                'artificial_bee_colony', 'slime_mould', 'sine_cosine'
            ])

        if PYSWARM_AVAILABLE:
            available.append('pyswarm_pso')

        if OPTUNA_AVAILABLE:
            available.append('optuna_bayesian')

        return available

    def _fitness_function(self, solution: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Fitness function for feature selection optimization.

        Parameters:
        -----------
        solution : np.ndarray
            Binary array indicating selected features
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector

        Returns:
        --------
        float
            Fitness score (higher is better)
        """
        # Convert solution to boolean mask
        if len(solution.shape) > 1:
            solution = solution.flatten()

        mask = solution > 0.5 if solution.dtype == float else solution.astype(bool)

        # Ensure at least one feature is selected
        if not np.any(mask):
            return 0.0

        try:
            # Select features
            X_selected = X[:, mask]

            # Use Random Forest for evaluation
            rf = RandomForestClassifier(n_estimators=10, random_state=self.random_state, n_jobs=-1)

            # Cross-validation score
            scores = cross_val_score(rf, X_selected, y, cv=3, scoring='accuracy', n_jobs=-1)
            fitness = np.mean(scores)

            # Penalize for too many features
            n_selected = np.sum(mask)
            penalty = 0.01 * (n_selected / len(mask))

            return fitness - penalty

        except Exception as e:
            self.logger.warning(f"Error in fitness evaluation: {str(e)}")
            return 0.0

    def _genetic_algorithm(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Genetic Algorithm for feature selection."""
        if ga is None:
            raise ImportError("geneticalgorithm package not installed")

        n_features = X.shape[1]

        def fitness_wrapper(solution):
            return -self._fitness_function(solution, X, y)  # GA minimizes

        # GA parameters
        varbound = np.array([[0, 1]] * n_features)
        vartype = np.array([['int']] * n_features)

        algorithm_param = {
            'max_num_iteration': self.max_iter,
            'population_size': self.population_size,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': None
        }

        model = ga(function=fitness_wrapper, dimension=n_features,
                   variable_type_mixed=vartype, variable_boundaries=varbound,
                   algorithm_parameters=algorithm_param)

        model.run()
        solution = model.output_dict['variable']

        return solution.astype(bool)

    def _particle_swarm(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Particle Swarm Optimization for feature selection."""
        if not MEALPY_AVAILABLE:
            raise ImportError("mealpy package not installed")

        n_features = X.shape[1]

        def fitness_wrapper(solution):
            return self._fitness_function(solution, X, y)

        # PSO parameters
        problem_dict = {
            "fit_func": fitness_wrapper,
            "lb": [0] * n_features,
            "ub": [1] * n_features,
            "minmax": "max",
        }

        model = PSO.OriginalPSO(epoch=self.max_iter, pop_size=self.population_size)
        best_position, best_fitness = model.solve(problem_dict)

        return (best_position > 0.5).astype(bool)

    def _ant_colony(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ant Colony Optimization approximation using ABC."""
        if not MEALPY_AVAILABLE:
            raise ImportError("mealpy package not installed")

        return self._artificial_bee_colony(X, y)

    def _simulated_annealing(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Simulated Annealing for feature selection."""
        if not MEALPY_AVAILABLE:
            raise ImportError("mealpy package not installed")

        n_features = X.shape[1]

        def fitness_wrapper(solution):
            return self._fitness_function(solution, X, y)

        problem_dict = {
            "fit_func": fitness_wrapper,
            "lb": [0] * n_features,
            "ub": [1] * n_features,
            "minmax": "max",
        }

        model = SA.OriginalSA(epoch=self.max_iter, pop_size=self.population_size)
        best_position, best_fitness = model.solve(problem_dict)

        return (best_position > 0.5).astype(bool)

    def _whale_optimization(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Whale Optimization Algorithm for feature selection."""
        if not MEALPY_AVAILABLE:
            raise ImportError("mealpy package not installed")

        n_features = X.shape[1]

        def fitness_wrapper(solution):
            return self._fitness_function(solution, X, y)

        problem_dict = {
            "fit_func": fitness_wrapper,
            "lb": [0] * n_features,
            "ub": [1] * n_features,
            "minmax": "max",
        }

        model = WOA.OriginalWOA(epoch=self.max_iter, pop_size=self.population_size)
        best_position, best_fitness = model.solve(problem_dict)

        return (best_position > 0.5).astype(bool)

    def _grey_wolf_optimization(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Grey Wolf Optimizer for feature selection."""
        if not MEALPY_AVAILABLE:
            raise ImportError("mealpy package not installed")

        n_features = X.shape[1]

        def fitness_wrapper(solution):
            return self._fitness_function(solution, X, y)

        problem_dict = {
            "fit_func": fitness_wrapper,
            "lb": [0] * n_features,
            "ub": [1] * n_features,
            "minmax": "max",
        }

        model = GWO.OriginalGWO(epoch=self.max_iter, pop_size=self.population_size)
        best_position, best_fitness = model.solve(problem_dict)

        return (best_position > 0.5).astype(bool)

    def _differential_evolution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Differential Evolution for feature selection."""
        if not MEALPY_AVAILABLE:
            raise ImportError("mealpy package not installed")

        n_features = X.shape[1]

        def fitness_wrapper(solution):
            return self._fitness_function(solution, X, y)

        problem_dict = {
            "fit_func": fitness_wrapper,
            "lb": [0] * n_features,
            "ub": [1] * n_features,
            "minmax": "max",
        }

        model = DE.OriginalDE(epoch=self.max_iter, pop_size=self.population_size)
        best_position, best_fitness = model.solve(problem_dict)

        return (best_position > 0.5).astype(bool)

    def _artificial_bee_colony(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Artificial Bee Colony for feature selection."""
        if not MEALPY_AVAILABLE:
            raise ImportError("mealpy package not installed")

        n_features = X.shape[1]

        def fitness_wrapper(solution):
            return self._fitness_function(solution, X, y)

        problem_dict = {
            "fit_func": fitness_wrapper,
            "lb": [0] * n_features,
            "ub": [1] * n_features,
            "minmax": "max",
        }

        model = ABC.OriginalABC(epoch=self.max_iter, pop_size=self.population_size)
        best_position, best_fitness = model.solve(problem_dict)

        return (best_position > 0.5).astype(bool)

    def _slime_mould_algorithm(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Slime Mould Algorithm for feature selection."""
        if not MEALPY_AVAILABLE:
            raise ImportError("mealpy package not installed")

        n_features = X.shape[1]

        def fitness_wrapper(solution):
            return self._fitness_function(solution, X, y)

        problem_dict = {
            "fit_func": fitness_wrapper,
            "lb": [0] * n_features,
            "ub": [1] * n_features,
            "minmax": "max",
        }

        model = SMA.OriginalSMA(epoch=self.max_iter, pop_size=self.population_size)
        best_position, best_fitness = model.solve(problem_dict)

        return (best_position > 0.5).astype(bool)

    def _sine_cosine_algorithm(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Sine Cosine Algorithm for feature selection."""
        if not MEALPY_AVAILABLE:
            raise ImportError("mealpy package not installed")

        n_features = X.shape[1]

        def fitness_wrapper(solution):
            return self._fitness_function(solution, X, y)

        problem_dict = {
            "fit_func": fitness_wrapper,
            "lb": [0] * n_features,
            "ub": [1] * n_features,
            "minmax": "max",
        }

        model = SCA.OriginalSCA(epoch=self.max_iter, pop_size=self.population_size)
        best_position, best_fitness = model.solve(problem_dict)

        return (best_position > 0.5).astype(bool)

    def _optuna_bayesian(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Optuna Bayesian Optimization for feature selection."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna package not installed")

        n_features = X.shape[1]

        def objective(trial):
            # Create binary mask
            mask = np.array([trial.suggest_categorical(f'feature_{i}', [True, False])
                             for i in range(n_features)])

            if not np.any(mask):
                return 0.0

            return self._fitness_function(mask, X, y)

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.max_iter * self.population_size)

        # Extract best solution
        best_params = study.best_params
        mask = np.array([best_params[f'feature_{i}'] for i in range(n_features)])

        return mask.astype(bool)

    def _random_search(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Random Search for feature selection (baseline)."""
        n_features = X.shape[1]
        np.random.seed(self.random_state)

        best_score = -np.inf
        best_mask = None

        n_trials = self.max_iter * self.population_size

        for _ in range(n_trials):
            # Generate random binary mask
            if self.n_features:
                mask = np.zeros(n_features, dtype=bool)
                selected_indices = np.random.choice(n_features, size=self.n_features, replace=False)
                mask[selected_indices] = True
            else:
                # Random number of features between 1 and n_features
                n_select = np.random.randint(1, n_features + 1)
                mask = np.random.choice([True, False], size=n_features,
                                        p=[n_select / n_features, 1 - n_select / n_features])

                # Ensure at least one feature is selected
                if not np.any(mask):
                    mask[np.random.randint(n_features)] = True

            score = self._fitness_function(mask, X, y)

            if score > best_score:
                best_score = score
                best_mask = mask.copy()

        return best_mask

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MetaHeuristicFeatureSelector':
        """
        Fit the feature selector to the data.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector

        Returns:
        --------
        self
        """
        self.logger.info(f"Starting feature selection with {self.algorithm}")

        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]

        # Check if algorithm is available
        available_algorithms = self.get_available_algorithms()
        if self.algorithm not in available_algorithms:
            self.logger.warning(f"Algorithm {self.algorithm} not available. Using random_search instead.")
            self.algorithm = 'random_search'

        # Set default number of features if not specified
        if self.n_features is None:
            self.n_features = min(int(X.shape[1] * 0.7), X.shape[1] - 1)
            self.logger.info(f"Auto-selected number of features: {self.n_features}")

        # Run the selected algorithm
        try:
            algorithm_func = self.available_algorithms[self.algorithm]
            self.selected_features_ = algorithm_func(X, y)

            n_selected = np.sum(self.selected_features_)
            self.logger.info(f"Selected {n_selected} features out of {X.shape[1]} using {self.algorithm}")

            # Evaluate the selected features
            if n_selected > 0:
                X_selected = X[:, self.selected_features_]
                rf = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
                scores = cross_val_score(rf, X_selected, y, cv=5, scoring='accuracy')
                self.logger.info(
                    f"Cross-validation accuracy with selected features: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

        except Exception as e:
            self.logger.error(f"Error in {self.algorithm}: {str(e)}")
            self.logger.info("Falling back to random search")
            self.selected_features_ = self._random_search(X, y)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data by selecting features.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix to transform

        Returns:
        --------
        np.ndarray
            Transformed feature matrix
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet.")

        if hasattr(X, 'values'):
            X = X.values

        return X[:, self.selected_features_]

    def get_selected_features(self) -> List[str]:
        """Get the names of selected features."""
        if self.selected_features_ is None or self.feature_names_ is None:
            return []

        return [name for name, selected in zip(self.feature_names_, self.selected_features_) if selected]


def load_intel_yaml(intel_path: str) -> Dict[str, Any]:
    """Load intelligence from YAML file."""
    logger = get_logger(__name__)

    try:
        with open(intel_path, 'r') as file:
            intel = yaml.safe_load(file)
        logger.info(f"Successfully loaded intel from {intel_path}")
        return intel
    except Exception as e:
        logger.error(f"Error loading intel file: {str(e)}")
        raise


def save_intel_yaml(intel: Dict[str, Any], intel_path: str) -> None:
    """Save intelligence to YAML file."""
    logger = get_logger(__name__)

    try:
        with open(intel_path, 'w') as file:
            yaml.dump(intel, file, default_flow_style=False, sort_keys=False)
        logger.info(f"Successfully saved intel to {intel_path}")
    except Exception as e:
        logger.error(f"Error saving intel file: {str(e)}")
        raise


def feature_selection_pipeline(intel_path: str = "intel.yaml",
                               algorithm: str = "genetic_algorithm",
                               n_features: Optional[int] = None,
                               max_iter: int = 50,
                               population_size: int = 20) -> None:
    """
    Execute the feature selection pipeline.

    Parameters:
    -----------
    intel_path : str
        Path to the intel.yaml file
    algorithm : str
        Meta-heuristic algorithm to use
    n_features : int or None
        Number of features to select
    max_iter : int
        Maximum iterations for the algorithm
    population_size : int
        Population size for the algorithm
    """
    logger = get_logger(__name__)

    section("FEATURE SELECTION PIPELINE", logger)

    try:
        # Load intelligence
        intel = load_intel_yaml(intel_path)

        # Extract required information
        dataset_name = intel['dataset_name']
        processor_pipeline_path = intel['processor_pipeline_path']
        target_column = intel['target_column']
        train_transformed_path = intel['train_transformed_path']
        test_transformed_path = intel['test_transformed_path']

        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"Target column: {target_column}")
        logger.info(f"Algorithm: {algorithm}")

        # Load transformed data
        logger.info("Loading transformed training data...")
        train_df = pd.read_csv(train_transformed_path)

        logger.info("Loading transformed test data...")
        test_df = pd.read_csv(test_transformed_path)

        # Separate features and target
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")

        # Initialize feature selector
        logger.info(f"Initializing {algorithm} feature selector...")
        selector = MetaHeuristicFeatureSelector(
            algorithm=algorithm,
            n_features=n_features,
            max_iter=max_iter,
            population_size=population_size,
            random_state=42
        )

        # Show available algorithms
        available_algs = selector.get_available_algorithms()
        logger.info(f"Available algorithms: {', '.join(available_algs)}")

        # Fit the selector
        logger.info("Fitting feature selector...")
        selector.fit(X_train, y_train)

        # Transform the data
        logger.info("Transforming training and test data...")
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)

        # Get selected feature names
        selected_features = selector.get_selected_features()
        logger.info(f"Selected features: {selected_features}")

        # Create selected dataframes
        train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
        train_selected_df[target_column] = y_train.values

        test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)
        test_selected_df[target_column] = y_test.values

        # Create output directories
        selected_dir = f"data/selected/data_{dataset_name}"
        os.makedirs(selected_dir, exist_ok=True)

        pipeline_dir = os.path.dirname(processor_pipeline_path)
        os.makedirs(pipeline_dir, exist_ok=True)

        # Save selected data
        train_selected_path = os.path.join(selected_dir, "train_selected.csv")
        test_selected_path = os.path.join(selected_dir, "test_selected.csv")

        train_selected_df.to_csv(train_selected_path, index=False)
        test_selected_df.to_csv(test_selected_path, index=False)

        logger.info(f"Saved selected training data to: {train_selected_path}")
        logger.info(f"Saved selected test data to: {test_selected_path}")

        # Save selection pipeline
        selection_pipeline_path = os.path.join(pipeline_dir, "selection.pkl")

        with open(selection_pipeline_path, 'wb') as f:
            cloudpickle.dump(selector, f)

        logger.info(f"Saved selection pipeline to: {selection_pipeline_path}")

        # Load existing processor pipeline
        logger.info("Loading existing processor pipeline...")
        with open(processor_pipeline_path, 'rb') as f:
            processor_pipeline = cloudpickle.load(f)

        # Create combined pipeline
        logger.info("Creating combined pipeline (processor + selector)...")
        combined_pipeline = Pipeline([
            ('processor', processor_pipeline),
            ('selector', selector)
        ])

        # Save combined pipeline as new processor.pkl
        with open(processor_pipeline_path, 'wb') as f:
            cloudpickle.dump(combined_pipeline, f)

        logger.info(f"Saved combined pipeline to: {processor_pipeline_path}")

        # Update intel.yaml
        logger.info("Updating intel.yaml...")
        intel.update({
            'selection_pipeline_path': selection_pipeline_path,
            'train_selected_path': train_selected_path,
            'test_selected_path': test_selected_path,
            'feature_selection_config': {
                'algorithm': algorithm,
                'n_features_selected': len(selected_features),
                'selected_features': selected_features,
                'max_iter': max_iter,
                'population_size': population_size,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        })

        save_intel_yaml(intel, intel_path)

        section("FEATURE SELECTION COMPLETED SUCCESSFULLY", logger)
        logger.info(f"Selected {len(selected_features)} features using {algorithm}")
        logger.info(f"Training data shape after selection: {X_train_selected.shape}")
        logger.info(f"Test data shape after selection: {X_test_selected.shape}")

    except Exception as e:
        logger.error(f"Error in feature selection pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage with user input
    logger = get_logger(__name__)

    # Get available algorithms
    selector = MetaHeuristicFeatureSelector()
    available_algorithms = selector.get_available_algorithms()

    print("\nAvailable Meta-Heuristic Algorithms:")
    for i, alg in enumerate(available_algorithms, 1):
        print(f"{i}. {alg}")

    try:
        # User algorithm selection
        choice = input(f"\nSelect algorithm (1-{len(available_algorithms)}): ")
        algorithm_idx = int(choice) - 1

        if algorithm_idx < 0 or algorithm_idx >= len(available_algorithms):
            raise ValueError("Invalid choice")

        selected_algorithm = available_algorithms[algorithm_idx]

        # User feature count selection
        n_features_input = input("\nNumber of features to select (press Enter for auto): ")
        n_features = int(n_features_input) if n_features_input.strip() else None

        # Run pipeline
        feature_selection_pipeline(
            intel_path="intel.yaml",
            algorithm=selected_algorithm,
            n_features=n_features,
            max_iter=50,
            population_size=20
        )

    except KeyboardInterrupt:
        logger.info("Feature selection cancelled by user")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        # Fallback to random search
        logger.info("Running with default settings (random_search)...")
        feature_selection_pipeline(
            intel_path="intel.yaml",
            algorithm="random_search"
        )