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
    A class for automatic hyperparameter tuning and training of a regression
    model.
    RH 2023
    
    Attributes:
        model_class (Type[sklearn.base.BaseEstimator]):
            A Scikit-Learn estimator class.
            Must have: \n
                * Method: ``fit(X, y)``
                * Method: ``predict_proba(X)`` \n
        params (Dict[str, Dict[str, Any]]):
            A dictionary of hyperparameters with their names, types, and bounds.
        X (np.ndarray):
            Input data.
            Shape: *(n_samples, n_features)*
        y (np.ndarray):
            Output data.
            Shape: *(n_samples,)*
        cv (Type[sklearn.model_selection._split.BaseCrossValidator]):
            A Scikit-Learn cross-validator class.
            Must have: \n
                * Call signature: ``idx_train, idx_test = next(self.cv.split(self.X, self.y))``
        fn_loss (Callable):
            Function to compute the loss.
            Must have: \n
                * Call signature: ``loss, loss_train, loss_test = fn_loss(y_pred_train, y_pred_test, y_true_train, y_true_test, sample_weight_train, sample_weight_test)`` \n
        n_jobs_optuna (int):
            Number of jobs for Optuna. Set to ``-1`` to use all cores.
            Note that some ``'solver'`` options are already parallelized (like
            ``'lbfgs'``). Set ``n_jobs_optuna`` to ``1`` for these solvers.
        n_startup (int):
            The number of startup trials for the optuna pruner and sampler.
        kwargs_convergence (Dict[str, Union[int, float]]):
            Convergence settings for the optimization. Includes: \n
                * ``'n_patience'`` (int): The number of trials to wait for convergence before stopping the optimization.
                * ``'tol_frac'`` (float): The fractional tolerance for convergence. After ``n_patience`` trials, the optimization will stop if the loss has not improved by at least ``tol_frac``.
                * ``'max_trials'`` (int): The maximum number of trials to run.
                * ``'max_duration'`` (int): The maximum duration of the optimization in seconds. \n
        sample_weight (Optional[np.ndarray]):
            Weights for the samples, equal to ones_like(y) if None.
        catch_convergence_warnings (bool):
            If ``True``, ignore ConvergenceWarning during model fitting.
        verbose (bool):
            If ``True``, show progress bar and print running results.

    """
    def __init__(
        self, 
        model_class: Type[sklearn.base.BaseEstimator], 
        params: Dict[str, Dict[str, Any]], 
        X: Any, 
        y: Any, 
        cv: Any, 
        fn_loss: Callable, 
        n_jobs_optuna: int = -1,
        n_startup: int = 15,
        kwargs_convergence = {
            'n_patience': 100,
            'tol_frac': 0.05,
            'max_trials': 350,
            'max_duration': 60*10,
        }, 
        sample_weight: Optional[Any] = None, 
        catch_convergence_warnings: bool = True,
        verbose=True,
    ):
        """
        Initializes the AutotunerRegression with the given model class, parameters, data, and settings.
        """
        ## Set model variables
        self.X = X  ## shape (n_samples, n_features)
        self.y = y  ## shape (n_samples,)
        self.model_class = model_class  ## sklearn estimator class
        self.sample_weight = sample_weight if sample_weight is not None else np.ones_like(self.y)
        self.cv = cv  ## sklearn cross-validator object with split method

        ## Set optuna variables
        self.n_startup = n_startup
        self.params = params
        self.n_jobs_optuna = n_jobs_optuna
        self.fn_loss = fn_loss
        self.catch_convergence_warnings = catch_convergence_warnings


        self.kwargs_convergence = kwargs_convergence

        self.verbose = verbose

        # Initialize a convergence checker
        self.checker = helpers.Convergence_checker_optuna(verbose=False, **self.kwargs_convergence)

        # Initialize variables to store loss and best model
        self.loss_running_train = []
        self.loss_running_test = []
        self.loss_running = []
        self.model_best = None
        self.loss_best = np.inf
        self.params_best = None

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """
        Define the objective function for Optuna to optimize.

        Args:
            trial (optuna.trial.Trial): 
                An Optuna trial object.

        Returns:
            (float): 
                loss (float):
                    The score of the classifier.
        """
        # Make a lookup table for the suggest methods
        LUT_suggest = {
            'categorical': trial.suggest_categorical,
            'real': trial.suggest_float,
            'int': trial.suggest_int,
            'discrete_uniform': trial.suggest_discrete_uniform,
            'loguniform': trial.suggest_loguniform,
            'uniform': trial.suggest_uniform,
        }

        # Suggest hyperparameters using optuna
        kwargs_model = {}
        for name, config in self.params.items():
            kwargs_model[name] = LUT_suggest[config['type']](name, **config['kwargs'])

        # Initialize the classifier with the suggested hyperparameters
        model = self.model_class(**kwargs_model)

        # Split the data
        idx_train, idx_test = next(self.cv.split(self.X, self.y))
        X_train, y_train, X_test, y_test = self.X[idx_train], self.y[idx_train], self.X[idx_test], self.y[idx_test]
        sample_weight_train, sample_weight_test = self.sample_weight[idx_train], self.sample_weight[idx_test]

        # Train the classifier
        ## Turn off ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=sklearn.exceptions.ConvergenceWarning) if self.catch_convergence_warnings else None
            model.fit(X_train, y_train)

        # Transform the training data
        y_pred_train = model.predict_proba(X_train)
        y_pred_test = model.predict_proba(X_test)

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
        self.loss_running.append(loss)

        # Update the bests
        if loss < self.loss_best:
            self.loss_best = loss
            self.model_best = model
            self.params_best = kwargs_model

        return loss

    def fit(self) -> Union[sklearn.base.BaseEstimator, Optional[Dict[str, Any]]]:
        """
        Fit and tune the hyperparameters and train the classifier.

        Returns:
            (Union[sklearn.base.BaseEstimator, Optional[Dict[str, Any]]): 
                best_model (sklearn.base.BaseEstimator):
                    The best estimator obtained from hyperparameter tuning.
                best_params (Optional[Dict[str, Any]]):
                    The best parameters obtained from hyperparameter tuning.
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
            n_jobs=self.n_jobs_optuna, 
            callbacks=[self.checker.check], 
            n_trials=self.kwargs_convergence['max_trials'],
            show_progress_bar=self.verbose,
        )        

        # Retrieve the best parameters and the best classifier
        self.best_params = self.study.best_params
        self.model_best = self.model_class(**self.best_params)
        
        # Train the classifier on the full data set
        self.model_best.fit(self.X, self.y)

        return self.model_best, self.best_params
    

class LossFucntion_CrossEntropy_CV():
    """
    Calculates the cross-entropy loss of a classifier using cross-validation. 
    RH 2023

    Args:
        penalty_testTrainRatio (float): 
            The amount of penalty for the test loss to the train loss. 
            Penalty is applied with formula: 
            ``loss = loss_test_or_train * ((loss_test / loss_train) ** penalty_testTrainRatio)``.
        labels (Optional[Union[List, np.ndarray]]): 
            A list or ndarray of labels. 
            Shape: *(n_samples,)*.
        test_or_train (str): 
            A string indicating whether to apply the penalty to the test or
            train loss.
            It should be either ``'test'`` or ``'train'``. 
    """
    def __init__(
        self,
        penalty_testTrainRatio: float = 1.0,
        labels: Optional[Union[List, np.ndarray]] = None,
        test_or_train: str = 'test',
    ) -> None:
        """
        Initializes the LossFunctionCrossEntropyCV with the given penalty, labels, and test_or_train setting.
        """
        self.labels = labels
        self.penalty_testTrainRatio = penalty_testTrainRatio
        ## Set the penalty function
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
        """
        Calculates the cross-entropy loss using cross-validation.

        Args:
            y_pred_train (np.ndarray): 
                Predicted output data for the training set. (shape:
                *(n_samples,)*)
            y_pred_test (np.ndarray): 
                Predicted output data for the test set. (shape: *(n_samples,)*)
            y_train_true (np.ndarray): 
                True output data for the training set. (shape: *(n_samples,)*)
            y_test_true (np.ndarray): 
                True output data for the test set. (shape: *(n_samples,)*)
            sample_weight_train (Optional[List[float]]): 
                Weights assigned to each training sample. 
            sample_weight_test (Optional[List[float]]): 
                Weights assigned to each test sample.

        Returns:
            (tuple): tuple containing:
                loss (float): 
                    The calculated loss after applying the penalty.
                loss_train (float): 
                    The cross-entropy loss of the training set.
                loss_test (float): 
                    The cross-entropy loss of the test set.
        """
        # Calculate the cross-entropy loss using cross-validation.
        from sklearn.metrics import log_loss
        loss_train = log_loss(y_train_true, y_pred_train, sample_weight=sample_weight_train, labels=self.labels)
        loss_test =  log_loss(y_test_true,  y_pred_test,  sample_weight=sample_weight_test,  labels=self.labels)
        loss = self.fn_penalty_testTrainRation(loss_test, loss_train)

        return loss, loss_train, loss_test


class Auto_LogisticRegression(Autotuner_regression):
    """
    Implements automatic hyperparameter tuning for Logistic Regression.

    Args:
        X (np.ndarray):
            Training data. (shape: *(n_samples, n_features)*)
        y (np.ndarray):
            Target variable. (shape: *(n_samples,)*)
        params_LogisticRegression (Dict):
            Dictionary of Logistic Regression parameters. 
            For each item in the dictionary if item is: \n
                * ``list``: The parameter is tuned. If the values are numbers,
                  then the list wil be the bounds [low, high] to search over. If
                  the values are strings, then the list will be the categorical
                  values to search over.
                * **not** a ``list``: The parameter is fixed to the given value. \n
            See `LogisticRegression
            <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
            for a full list of arguments.
        n_startup (int):
            Number of startup trials. (Default is *15*)
        kwargs_convergence (Dict[str, Union[int, float]]):
            Convergence settings for the optimization. Includes: \n
                * ``'n_patience'`` (int): The number of trials to wait for
                  convergence before stopping the optimization.
                * ``'tol_frac'`` (float): The fractional tolerance for
                  convergence. After ``n_patience`` trials, the optimization
                  will stop if the loss has not improved by at least
                  ``tol_frac``.
                * ``'max_trials'`` (int): The maximum number of trials to run.
                * ``'max_duration'`` (int): The maximum duration of the
                  optimization in seconds. \n
        n_jobs_optuna (int):
            Number of jobs for Optuna. Set to ``-1`` to use all cores.
            Note that some ``'solver'`` options are already parallelized (like
            ``'lbfgs'``). Set ``n_jobs_optuna`` to ``1`` for these solvers.
        penalty_testTrainRatio (float):
            Penalty ratio for test and train. 
        test_size (float):
            Test set ratio.
        class_weight (Union[Dict[str, float], str]):
            Weights associated with classes in the form of a dictionary or
            string. If given "balanced", class weights will be calculated.
            (Default is "balanced")
        sample_weight (Optional[List[float]]):
            Sample weights. See `LogisticRegression
            <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
            for more details.
        cv (Optional[sklearn.model_selection._split.BaseCrossValidator]):
            A Scikit-Learn cross-validator class.
            If not ``None``, then must have: \n
                * Call signature: ``idx_train, idx_test =
                  next(self.cv.split(self.X, self.y))`` \n
            If ``None``, then a StratifiedShuffleSplit cross-validator will be
            used.
        verbose (bool):
            Whether to print progress messages.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params_LogisticRegression: Dict = {
            'C': [1e-14, 1e3],
            'penalty': 'l2',
            'fit_intercept': True,
            'solver': 'lbfgs',
            'max_iter': 1000,
            'tol': 0.0001,
            'n_jobs': None,
            'l1_ratio': None,
            'warm_start': False,
        },
        n_startup: int = 15,
        kwargs_convergence: Dict = {
            'n_patience': 100,
            'tol_frac': 0.05,
            'max_trials': 350,
            'max_duration': 60*10,
        }, 
        n_jobs_optuna: int = 1,
        penalty_testTrainRatio: float = 1.0,
        test_size: float = 0.3,
        class_weight: Optional[Union[Dict[str, float], str]] = 'balanced',
        sample_weight: Optional[List[float]] = None,
        cv: Optional[sklearn.model_selection._split.BaseCrossValidator] = None,
        verbose: bool = True,
    ) -> None:
        """
        Initializes Auto_LogisticRegression with the given parameters and data.
        """
        ## Prepare class weights
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

        ## Prepare the loss function
        self.fn_loss = LossFucntion_CrossEntropy_CV(
            penalty_testTrainRatio=penalty_testTrainRatio,
            labels=y,
        )

        ## Prepare the cross-validation
        self.cv = sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
        )

        ## Prepare static kwargs for sklearn LogisticRegression
        kwargs_LogisticRegression = {key: val for key, val in params_LogisticRegression.items() if isinstance(val, list)==False}

        ## Prepare dynamic kwargs for optuna
        params_OptunaSuggest = {key: val for key, val in params_LogisticRegression.items() if isinstance(val, list)==True}
        ### Make a mapping from sklearn LogisticRegression kwargs to optuna suggest types and kwargs
        params = {
            'C':             {'type': 'real',        'kwargs': {'log': True} },
            'penalty':       {'type': 'categorical', 'kwargs': {}            },
            'fit_intercept': {'type': 'real',        'kwargs': {'bool': True}},
            'solver':        {'type': 'categorical', 'kwargs': {}            },
            'max_iter':      {'type': 'int',         'kwargs': {'log': True} },
            'tol':           {'type': 'real',        'kwargs': {'log': True} },
            'n_jobs':        {'type': 'int',         'kwargs': {'log': True} },
            'l1_ratio':      {'type': 'real',        'kwargs': {'log': False}},
            'warm_start':    {'type': 'real',        'kwargs': {'bool': True}},
        }
        ### Prune mapping to only include params in params_OptunaSuggest
        params = {key: val for key, val in params.items() if key in params_OptunaSuggest.keys()}
        ### Add kwargs to params
        for key, val in params_OptunaSuggest.items():
            assert key in params.keys(), f'key "{key}" not in params_metadata.keys().'
            if params[key]['type'] in ['real', 'int']:
                kwargs = ['low', 'high', 'step', 'log']
                params[key]['kwargs'] = {**params[key]['kwargs'], **{kwargs[ii]: val[ii] for ii in range(len(val))}}
            elif params[key]['type'] == 'categorical':
                params[key]['kwargs'] = {**params[key]['kwargs'], **{'choices': val}}
            else:
                raise ValueError(f'params_metadata[{key}]["type"] must be either "real", "int", or "categorical". This error should never be raised.')
            
        ## Prepare the classifier class
        self.classifier_class = functools.partial(
            sklearn.linear_model.LogisticRegression,
            class_weight=self.class_weight,
            **kwargs_LogisticRegression,
        )

        ## Initialize the Autotuner superclass
        super().__init__(
            model_class=self.classifier_class,
            params=params,
            X=X,
            y=y,
            n_startup=n_startup,
            kwargs_convergence=kwargs_convergence,
            n_jobs_optuna=n_jobs_optuna,
            cv=self.cv,
            fn_loss=self.fn_loss,
            catch_convergence_warnings=True,
            verbose=verbose,
        )