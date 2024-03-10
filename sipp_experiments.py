"""Experiments on SIPP 2014 data."""

import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

from sipp import load_sipp

from baselines import backward_baselines
from baselines import compute_results
from baselines import ScoreWrapper
from baselines import roc_curve_flipped
from baselines import list_aggregator
from baselines import select_model_from_results

from plotters import bar_plot_results
from plotters import roc_plot_results


def sipp_features_dict():

    demographic_features = ['AGE', 'GENDER', 'RACE', 'EDUCATION', 'MARITAL_STATUS',
                            'CITIZENSHIP_STATUS']

    return { 'Education' : ['EDUCATION'],
             'Race' : ['RACE'],
             'Education, Race' : ['EDUCATION', 'RACE'],
             'All demographic' : demographic_features }


def sipp_rocplot():
    """Plot ROC curves."""

    scaled_logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, tol=0.1))
    models = [RandomForestClassifier(), GradientBoostingClassifier(), scaled_logistic]
    models = [ScoreWrapper(model) for model in models]


    features_dict = sipp_features_dict()

    file_name = 'results/sipp_roc.json'

    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            results = json.loads(f.read())
    else:
        X, y = load_sipp()
        results = compute_results(X, y, features_dict, models, 
                                  score_function=roc_curve_flipped, 
                                  baseline=GradientBoostingRegressor(),
                                  num_seeds=8,
                                  aggregator=list_aggregator)
        with open(file_name, 'w') as f:
            f.write(json.dumps(results))

    model_names = ['Random Forest', 'Gradient Boost', 'Logistic Reg']

    plot_file_name = 'results/sipp_rocplot.pdf'
    roc_plot_results(results, model_names=model_names,
                     plot_file_name=plot_file_name)

    plot_file_name = 'results/sipp_rocplot_logreg.pdf'
    sub_results = select_model_from_results(results, ScoreWrapper(scaled_logistic))
    roc_plot_results(sub_results, model_names=['Logistic Reg'],
                     plot_file_name=plot_file_name)


def sipp_zeroone_barplot():
    """Bar plots for zero-one loss."""

    scaled_logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, tol=0.1))
    models = [RandomForestClassifier(), GradientBoostingClassifier(), scaled_logistic]

    features_dict = sipp_features_dict()

    file_name = 'results/sipp_zeroone.json'
        
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            results = json.loads(f.read())
    else:
        X, y = load_sipp()
        results = compute_results(X, y, features_dict, models, num_seeds=8)
        with open(file_name, 'w') as f:
            f.write(json.dumps(results))


    model_names = ['Random Forest', 'Gradient Boost', 'Logistic Reg']

    plot_file_name = 'results/sipp_zeroone_barplot.pdf'
    bar_plot_results(results, model_names=model_names,
                     constant_baseline=0.4888,
                     plot_file_name=plot_file_name)

    plot_file_name = 'results/sipp_zeroone_barplot_alltests.pdf'
    bar_plot_results(results, model_names=model_names,
                     constant_baseline=0.4888,
                     tests=['XYY', 'WYY', 'WY^Y', 'WYY^', 'WY^Y^'],
                     plot_file_name=plot_file_name)

    plot_file_name = 'results/sipp_zeroone_barplot_logreg.pdf'
    sub_results = select_model_from_results(results, scaled_logistic)
    bar_plot_results(sub_results, model_names=['Logistic Reg'], loss_name='zero-one loss', 
                     plot_file_name=plot_file_name,
                     constant_baseline=0.4888)


def sipp_squared_barplot():
    """Bar plots for squared loss."""

    scaled_logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, tol=0.1))
    models = [RandomForestClassifier(), GradientBoostingClassifier(), scaled_logistic]
    models = [ScoreWrapper(model) for model in models]

    features_dict = sipp_features_dict()

    file_name = 'results/sipp_squared.json'

    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            results = json.loads(f.read())
    else:
        X, y = load_sipp()
        results = compute_results(X, y, features_dict, models, 
                                  score_function=mean_squared_error, 
                                  baseline=GradientBoostingRegressor(),
                                  num_seeds=8)
        with open(file_name, 'w') as f:
            f.write(json.dumps(results))

    model_names = ['Random Forest', 'Gradient Boost', 'Logistic Reg']

    plot_file_name = 'results/sipp_squared_barplot.pdf'
    bar_plot_results(results, model_names=model_names, loss_name='squared loss',
                     constant_baseline=0.2498,
                     plot_file_name=plot_file_name)

    plot_file_name = 'results/sipp_squared_barplot_alltests.pdf'
    bar_plot_results(results, model_names=model_names, loss_name='squared loss',
                     constant_baseline=0.2498,
                     tests=['XYY', 'WYY', 'WY^Y', 'WYY^', 'WY^Y^'],
                     plot_file_name=plot_file_name)

    plot_file_name = 'results/sipp_squared_barplot_logreg.pdf'
    sub_results = select_model_from_results(results, ScoreWrapper(scaled_logistic))
    bar_plot_results(sub_results, model_names=['Logistic Reg'], loss_name='squared loss',
                     plot_file_name=plot_file_name,
                     constant_baseline=0.2498)


if __name__ == '__main__':

    sipp_rocplot()
    sipp_zeroone_barplot()
    sipp_squared_barplot()
