"""Methods for computing backward baselines."""

import numpy as np

from parallelize import parallelize

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve

from sklearn.ensemble import GradientBoostingClassifier


class ScoreWrapper:
    """Wrapper to turn classifiers into regressors."""
    
    def __init__(self, model):
        self._model = model
        self.fit = getattr(model, 'fit')
        
    def __str__(self):
        return "ScoreWrapper(" + str(self._model) + ")"
    
    def predict(self, X):
        return self._model.predict_proba(X)[:, 1]

    
def zero_one_loss(y1, y2):
    """Classification error."""
    return 1.0 - accuracy_score(y1 , y2)


def roc_curve_flipped(yhat, y):
    """Round and flip arguments"""
    y = 1.0 * (y > 0.5)
    fprs, tprs, thresholds = roc_curve(y, yhat)
    return (fprs.tolist(), tprs.tolist(), thresholds.tolist())


def backward_baselines(X, y, features, models, score_function=zero_one_loss,
                                               test_size=0.33,
                                               random_state=None,
                                               baseline=GradientBoostingClassifier()):
    """Compute backward baselines.

    Parameters
    ----------
    X : numpy.ndarray
        data matrix (n, d)
    y : numpy.ndarray
        target variable (n,)
    features : list
        list of column names
    models : list
        list of models supporting fit and predict
    score_function : function
        score function
    test_size : float
        fraction of data used for training
    baseline : object
        model supporting fit and predict
    
    Returns
    -------
    dict
        Scores of all backward baselines for each model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    
    results = {}
    for model in models:

        scores = {}

        # XYY
        model.fit(X_train, y_train)
        scores['XYY'] = score_function(model.predict(X_test), y_test)

        # WYY
        baseline.fit(X_train[features], y_train)
        scores['WYY'] = score_function(baseline.predict(X_test[features]), y_test)

        # WY^Y
        baseline.fit(X_test[features], model.predict(X_test))
        scores['WY^Y'] = score_function(baseline.predict(X_test[features]), y_test)

        # WYY^
        baseline.fit(X_test[features], y_test)
        scores['WYY^'] = score_function(baseline.predict(X_test[features]), model.predict(X_test))

        # WY^Y^ requires new train/test split
        X_testA, X_testB, y_testA, y_testB = train_test_split(X_test[features],
                                                  model.predict(X_test),
                                                  test_size=0.5)
        baseline.fit(X_testA, y_testA)
        scores['WY^Y^'] = score_function(baseline.predict(X_testB), y_testB)

        results[str(model)] = scores

    return results


def select_model_from_results(results, model):
    """Keep only one model from results."""
    sub_results = {}
    for feature in results:
        sub_results[feature] = {}
        for m in results[feature]:
            if str(m) == str(model):
                sub_results[feature][m] = results[feature][m]
    return sub_results


def print_results(results):
    """Print results returned by backward_baselines"""
    for model in results:
        print(model)
        for test in results[model]:
            print("  ", test, ": ", results[model][test])


def mean_std_aggregator(results_list):
    """Aggregate a list of results computed by backward_baselines."""
    res = {}
    results = results_list[0]
    for model in results.keys():
        res[model] = {}
        for test in results[model]:
            mean_score = np.mean([r[model][test] for r in results_list])
            std_score = np.std([r[model][test] for r in results_list])
            res[model][test] = { 'mean' : mean_score, 'std' : std_score }
    return res


def list_aggregator(results_list):
    """Aggregate results by keeping a list of values."""
    res = {}
    results = results_list[0]
    for model in results.keys():
        res[model] = {}
        for test in results[model]:
            res[model][test] = [r[model][test] for r in results_list]
    return res


def compute_results(X, y, features_dict, models, 
                    score_function=zero_one_loss,
                    test_size=0.33,
                    random_state=None,
                    baseline=GradientBoostingClassifier(),
                    num_seeds=10,
                    aggregator=mean_std_aggregator):
    """Run multiple executions of backward_baselines in parallel."""

    random_seeds = range(num_seeds)

    r = {}
    for features in features_dict:
        f = lambda s: backward_baselines(X, y, 
                                         features_dict[features],
                                         models,
                                         test_size=test_size,
                                         random_state=s,
                                         score_function=score_function,
                                         baseline=baseline)

        results_list = parallelize(random_seeds, f)
        r[features] = aggregator(results_list)

    return r

