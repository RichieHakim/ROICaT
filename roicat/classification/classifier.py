# Write a superclass that does automatic hyperparameter tuning and training of a classifier. The class should do the following:
# 1. Ingest:
#      a. An sklearn classifier class like LogisticRegression
#      b. A dictionary of hyperparameters with: Their names, types (categorical, real, etc.), and their bounds.
#      c. Input data: X, y
#      d. Parameters for the hyperparameter tuning: max_iter, max_time, n_jobs, cv class, scoring_function
# 2. Train the classifier using the hyperparameters and data
#      a. Suggest hyperparameters using optuna and initialize the classifier with them
#      b. Split the data into training and validation sets
#      c. Train the classifier on the training set
#      d. Evaluate the classifier using the scoring function
#           i. The scoring should be an input function that takes in: (estimator, X_train, y_train, X_val, y_val) and returns a loss value.
#      e. Repeat steps until convergence criteria is reached
# 3. Return the best classifier and its hyperparameters

# - Use optuna for the hyperparameter search and tuning backend.
# - Write documentation for each step, use type hints, Google-style docstrings, and line-by-line comments.

import warnings
from typing import Dict, Type, Any, Union, Optional, Callable, Tuple, List
import functools

import numpy as np

import optuna
import sklearn

from .. import helpers

class Autotuner_regression:
    """
    A class for automatic hyperparameter tuning and training of a classifier.

    Attributes:
        clf_class: A sklearn classifier class.
        params: A dictionary of hyperparameters with their names, types and their bounds.
        X: Input data.
        y: Output data.
        max_iter: The maximum number of iterations for the hyperparameter tuning.
        max_time: The maximum time for the hyperparameter tuning.
        n_jobs: The number of jobs to run in parallel.
        cv: A cross-validation class.
        scoring: The scoring method to evaluate the performance of the classifier.

    """

    def __init__(
        self, 
        model_class: Type[sklearn.base.BaseEstimator], 
        params: Dict[str, Dict[str, Any]], 
        X: Any, 
        y: Any, 
        cv: Any, 
        fn_loss: Callable, 
        n_jobs: int = -1,
        n_startup: int = 15,
        kwargs_findParameters = {
            'n_patience': 100,
            'tol_frac': 0.05,
            'max_trials': 350,
            'max_duration': 60*10,
            'verbose': False,
        }, 
        sample_weight: Optional[Any] = None, 
        catch_convergence_warnings: bool = True,
    ):
        self.model_class = model_class
        self.params = params
        self.X = X
        self.y = y
        self.n_jobs = n_jobs
        self.cv = cv
        self.fn_loss = fn_loss
        self.best_clf = None
        self.best_params = None
        self.sample_weight = sample_weight if sample_weight is not None else np.ones_like(self.y)

        self.catch_convergence_warnings = catch_convergence_warnings

        self.kwargs_findParameters = kwargs_findParameters
        self.n_startup = n_startup

        self.checker = helpers.Convergence_checker_optuna(**self.kwargs_findParameters)

        self.loss_running_train = []
        self.loss_running_test = []

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """
        Define the objective function for Optuna to optimize.

        Args:
            trial: An Optuna trial object.

        Returns:
            The score of the classifier.
        """

        # Suggest hyperparameters using optuna
        params = {}
        for (param, type_param), config in self.params.items():
            if type_param == 'categorical':
                params[param] = trial.suggest_categorical(param, **config)
            elif type_param == 'real':
                params[param] = trial.suggest_float(param, **config)
            elif type_param == 'int':
                params[param] = trial.suggest_int(param, **config)

        # Initialize the classifier with the suggested hyperparameters
        clf = self.model_class(**params)

        # Split the data
        idx_train, idx_test = next(self.cv.split(self.X, self.y))
        X_train, y_train, X_test, y_test = self.X[idx_train], self.y[idx_train], self.X[idx_test], self.y[idx_test]
        sample_weight_train, sample_weight_test = self.sample_weight[idx_train], self.sample_weight[idx_test]

        # Train the classifier
        ## Turn off ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=sklearn.exceptions.ConvergenceWarning) if self.catch_convergence_warnings else None
            clf.fit(X_train, y_train)

        # Transform the training data
        y_pred_train = clf.predict_proba(X_train)
        y_pred_test = clf.predict_proba(X_test)

        # Evaluate the classifier using the scoring method
        loss, loss_train, loss_test = self.fn_loss(
            y_pred_train=y_pred_train, 
            y_pred_test=y_pred_test,
            y_train_true=y_train,
            y_test_true=y_test,
            sample_weight_train=sample_weight_train,
            sample_weight_test=sample_weight_test,
        )

        # Save the running loss
        self.loss_running_train.append(loss_train)
        self.loss_running_test.append(loss_test)

        return loss

    def fit(self) -> Union[sklearn.base.BaseEstimator, Optional[Dict[str, Any]]]:
        """
        Fit and tune the hyperparameters and train the classifier.

        Returns:
            The best classifier and its hyperparameters.
        """

        # Initialize an Optuna study
        self.study = optuna.create_study(
            direction="minimize", 
            pruner=optuna.pruners.MedianPruner(n_startup_trials=self.n_startup), 
            sampler=optuna.samplers.TPESampler(n_startup_trials=self.n_startup),
            study_name='Autotuner',
        )

        # Optimize the study
        self.study.optimize(
            self._objective, 
            n_jobs=self.n_jobs, 
            callbacks=[self.checker.check], 
            n_trials=self.kwargs_findParameters['max_trials'],
            show_progress_bar=self.kwargs_findParameters['verbose']
        )        

        # Retrieve the best parameters and the best classifier
        self.best_params = self.study.best_params
        self.model_best = self.model_class(**self.best_params)
        
        # Train the classifier on the full data set
        self.model_best.fit(self.X, self.y)

        return self.model_best, self.best_params
    

class LossFucntion_CrossEntropy_CV():
    def __init__(
        self,
        penalty_testTrainRatio: float = 1.0,
        labels: Optional[Union[List, np.ndarray]] = None,
        test_or_train: str = 'test',
    ) -> float:
        """
        Calculate the cross-entropy loss of a classifier using cross-validation.

        Args:
            estimator: A classifier. Must have a predict_proba method.
            X: Input data.
            y: Output data.
            cv: A cross-validation class. Must have a split method.
            penalty_testTrainRatio: The amount of penalty for the test loss to the train loss. Penalty is applied with formula: loss = loss_test_or_train * ((loss_test / loss_train) ** penalty_testTrainRatio).
        """
        self.labels = labels
        self.penalty_testTrainRatio = penalty_testTrainRatio
        if test_or_train == 'test':
            self.fn_penalty_testTrainRation = lambda test, train: test * ((test  / train) ** self.penalty_testTrainRatio)
        elif test_or_train == 'train':
            self.fn_penalty_testTrainRation = lambda test, train: train * ((train / test) ** self.penalty_testTrainRatio)
        else:
            raise ValueError('test_or_train must be either "test" or "train".')

    
    def __call__(
        self,
        y_pred_train: np.ndarray, 
        y_pred_test: np.ndarray,
        y_train_true: np.ndarray,
        y_test_true: np.ndarray,
        sample_weight_train: Optional[List[float]] = None,
        sample_weight_test: Optional[List[float]] = None,
    ):
        # Calculate the cross-entropy loss using cross-validation.
        from sklearn.metrics import log_loss
        loss_train = log_loss(y_train_true, y_pred_train, sample_weight=sample_weight_train, labels=self.labels)
        loss_test =  log_loss(y_test_true,  y_pred_test,  sample_weight=sample_weight_test,  labels=self.labels)
        loss = self.fn_penalty_testTrainRation(loss_test, loss_train)

        return loss, loss_train, loss_test


class Auto_LogisticRegression(Autotuner):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C_bounds: Union[Tuple[float, float], List[float]] = [1e-15, 1e4],
        n_startup=15,
        n_patience: int = 100,
        tol_frac: float = 0.05,
        max_trials: int = 350,
        max_duration: float = 60*10,
        verbose: bool = False,
        class_weight: Optional[Dict[str, float]] = 'balanced',
        sample_weight: Optional[List[float]] = None,
    ):
        self.classes = np.unique(y)

        class_weight = sklearn.utils.class_weight.compute_class_weight(
            class_weight=class_weight,
            y=y,
            classes=self.classes,
        )
        self.class_weight = {c: cw for c, cw in zip(self.classes, class_weight)}
        sample_weight = sklearn.utils.class_weight.compute_sample_weight(
            class_weight=sample_weight, 
            y=y,
        )

        self.fn_loss = LossFucntion_CrossEntropy_CV(
            penalty_testTrainRatio=1.0,
            labels=y,
        )

        self.cv = sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.3,
        )

        self.classifier_class = functools.partial(
            sklearn.linear_model.LogisticRegression,
            penalty='l2',
            fit_intercept=True,
            class_weight=self.class_weight,
            solver='lbfgs',
            max_iter=1000,
            verbose=0,
            n_jobs=None,
            l1_ratio=None,
            warm_start=False,
        )

        self.C_bounds = C_bounds

        ## Initialize the autotuner
        super().__init__(
            model_class=self.classifier_class,
            params={('C', 'real'): {'low': self.C_bounds[0], 'high': self.C_bounds[1], 'log': True}},
            X=X,
            y=y,
            n_startup=n_startup,
            kwargs_findParameters = {
                'n_patience': n_patience,
                'tol_frac': tol_frac,
                'max_trials': max_trials,
                'max_duration': max_duration,
                'verbose': verbose,
            },
            n_jobs=1,
            cv=self.cv,
            fn_loss=self.fn_loss,
            catch_convergence_warnings=True,
        )