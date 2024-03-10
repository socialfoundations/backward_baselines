"""Experiments on MEPS data."""

import os
import json

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from meps import load_meps

from baselines import roc_curve_flipped
from baselines import backward_baselines
from baselines import compute_results
from baselines import ScoreWrapper
from baselines import list_aggregator
from baselines import select_model_from_results

from plotters import bar_plot_results
from plotters import roc_plot_results


def meps_features_dict(): 
    demographic_columns = ['AGE31X', 'SEX','RACEV1X','RACEV2X','RACEAX','RACEBX','RACEWX','RACETHX',
                           'HISPANX','HISPNCAT','EDUCYR','HIDEG','OTHLGSPK','HWELLSPK','BORNUSA',
                           'WHTLGSPK','YRSINUS']

    features_dict = { 'Age' : ['AGE31X'],
                'Race' : ['RACETHX'],
                'Age, Race' : ['AGE31X', 'RACETHX'],
                'All demographic' : demographic_columns }

    return features_dict


def meps_rocplot():
    """Plot ROC curves."""


    scaled_logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, tol=0.1))
    models = [RandomForestClassifier(), GradientBoostingClassifier(), scaled_logistic]
    models = [ScoreWrapper(model) for model in models]

    features_dict = meps_features_dict()
    file_name = 'results/meps_roc.json'
        
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            results = json.loads(f.read())
    else:
        X, y = load_meps()
        results = compute_results(X, y, features_dict, models, 
                                  score_function=roc_curve_flipped, 
                                  baseline=GradientBoostingRegressor(),
                                  num_seeds=8,
                                  aggregator=list_aggregator)
        with open(file_name, 'w') as f:
            f.write(json.dumps(results))

    model_names = ['Random Forest', 'Gradient Boost', 'Logistic Reg']

    plot_file_name = 'results/meps_rocplot.pdf'
    roc_plot_results(results, model_names=model_names,
                     plot_file_name=plot_file_name)

    plot_file_name = 'results/meps_rocplot_logreg.pdf'
    sub_results = select_model_from_results(results, ScoreWrapper(scaled_logistic))
    roc_plot_results(sub_results, model_names=['Logistic Reg'],
                     plot_file_name=plot_file_name)


def meps_squared_barplot():
    """Bar plots for squared loss."""

    scaled_logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, tol=0.1))
    models = [RandomForestClassifier(), GradientBoostingClassifier(), scaled_logistic]
    models = [ScoreWrapper(model) for model in models]

    features_dict = meps_features_dict()
    file_name = 'results/meps_squared.json'
        
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            results = json.loads(f.read())
    else:
        X, y = load_meps()
        results = compute_results(X, y, features_dict, models, 
                                  score_function=mean_squared_error, 
                                  baseline=GradientBoostingRegressor(),
                                  num_seeds=8)
        with open(file_name, 'w') as f:
            f.write(json.dumps(results))

    model_names = ['Random Forest', 'Gradient Boost', 'Logistic Reg']

    plot_file_name = 'results/meps_squared_barplot.pdf'
    bar_plot_results(results, model_names=model_names, loss_name='squared loss',
                     plot_file_name=plot_file_name,
                     constant_baseline=0.2489)

    plot_file_name = 'results/meps_squared_barplot_alltests.pdf'
    bar_plot_results(results, model_names=model_names, loss_name='squared loss',
                     plot_file_name=plot_file_name,
                     tests=['XYY', 'WYY', 'WY^Y', 'WYY^', 'WY^Y^'],
                     constant_baseline=0.2489)

    plot_file_name = 'results/meps_squared_barplot_logreg.pdf'
    sub_results = select_model_from_results(results, ScoreWrapper(scaled_logistic))
    bar_plot_results(sub_results, model_names=['Logistic Reg'], loss_name='squared loss',
                     plot_file_name=plot_file_name,
                     constant_baseline=0.2489)


def meps_zeroone_barplot():
    """Bar plots for zero-one loss."""


    scaled_logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, tol=0.1))
    models = [RandomForestClassifier(), GradientBoostingClassifier(), scaled_logistic]

    features_dict = meps_features_dict()

    file_name = 'results/meps_zeroone.json'
        
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            results = json.loads(f.read())
    else:
        X, y = load_meps()
        results = compute_results(X, y, features_dict, models, num_seeds=8)
        with open(file_name, 'w') as f:
            f.write(json.dumps(results))

    model_names = ['Random Forest', 'Gradient Boost', 'Logistic Reg']

    plot_file_name = 'results/meps_zeroone_barplot.pdf'
    bar_plot_results(results, model_names=model_names, loss_name='zero-one loss', 
                     plot_file_name=plot_file_name,
                     constant_baseline=0.4682)

    plot_file_name = 'results/meps_zeroone_barplot_alltests.pdf'
    bar_plot_results(results, model_names=model_names, loss_name='zero-one loss', 
                     plot_file_name=plot_file_name,
                     tests=['XYY', 'WYY', 'WY^Y', 'WYY^', 'WY^Y^'],
                     constant_baseline=0.4682)

    plot_file_name = 'results/meps_zeroone_barplot_logreg.pdf'
    sub_results = select_model_from_results(results, scaled_logistic)
    bar_plot_results(sub_results, model_names=['Logistic Reg'], loss_name='zero-one loss', 
                     plot_file_name=plot_file_name,
                     constant_baseline=0.4682)


if __name__ == '__main__':

    meps_rocplot()
    meps_squared_barplot()
    meps_zeroone_barplot()
