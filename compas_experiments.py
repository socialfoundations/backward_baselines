import os
import json

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from plotters import bar_plot_results
from plotters import roc_plot_results

from baselines import backward_baselines
from baselines import compute_results
from baselines import roc_curve_flipped
from baselines import list_aggregator
from compas import load_compas
from compas import CompasModel
from compas import CompasRegressor


def compas_features_dict():
    return { 'Race' : ['raceint'],
             'Race, Age' : ['raceint', 'age'],
             'Juvenile priors' : ['juv_fel_count', 'juv_misd_count', 'juv_other_count'],
             'Priors count' : ['priors_count'] }


def compas_rocplot():
    """Create roc plot for Compas dataset."""

    X, y = load_compas()
    compas_model = CompasRegressor(X)
    models=[compas_model]

    features_dict = compas_features_dict()

    file_name = 'results/compas_roc.json'
        
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            results = json.loads(f.read())
    else:
        results = compute_results(X, y, features_dict, models, 
                                  score_function=roc_curve_flipped, 
                                  baseline=GradientBoostingRegressor(),
                                  num_seeds=8,
                                  aggregator=list_aggregator)
        with open(file_name, 'w') as f:
            f.write(json.dumps(results))

    model_names = ['Compas score']
    plot_file_name = 'results/compas_rocplot.pdf'
    roc_plot_results(results, model_names=model_names,
                     plot_file_name=plot_file_name)


def compas_zeroone_barplot():
    """Create barplot for Compas dataset with zero-one loss."""

    X, y = load_compas()
    compas_model = CompasModel(X)
    models=[compas_model]

    features_dict = compas_features_dict()

    file_name = 'results/compas_zeroone.json'
        
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            results = json.loads(f.read())
    else:
        results = compute_results(X, y, features_dict, models, num_seeds=8)
        with open(file_name, 'w') as f:
            f.write(json.dumps(results))

    model_names = ['Compas score']
    plot_file_name = 'results/compas_zeroone_barplot.pdf'
    bar_plot_results(results, model_names=model_names, loss_name='zero-one loss',
                     plot_file_name=plot_file_name,
                     constant_baseline=0.45065151095092876)

    plot_file_name = 'results/compas_zeroone_barplot_alltests.pdf'
    bar_plot_results(results, model_names=model_names, loss_name='zero-one loss',
                     plot_file_name=plot_file_name,
                     tests=['XYY', 'WYY', 'WY^Y', 'WYY^', 'WY^Y^'],
                     constant_baseline=0.45065151095092876)


def compas_squared_barplot():
    """Create barplot for Compas dataset with squared loss."""

    X, y = load_compas()
    compas_model = CompasRegressor(X)
    models=[compas_model]

    features_dict = compas_features_dict()

    file_name = 'results/compas_squared.json'
        
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            results = json.loads(f.read())
    else:
        results = compute_results(X, y, features_dict, models, 
                                  score_function=mean_squared_error, 
                                  baseline=GradientBoostingRegressor(),
                                  num_seeds=8)
        with open(file_name, 'w') as f:
            f.write(json.dumps(results))

    model_names = ['Compas score']
    plot_file_name = 'results/compas_squared_barplot.pdf'
    bar_plot_results(results, model_names=model_names, loss_name='squared loss',
                     plot_file_name=plot_file_name,
                     constant_baseline=0.2475647266285736)

    plot_file_name = 'results/compas_squared_barplot_alltests.pdf'
    bar_plot_results(results, model_names=model_names, loss_name='squared loss',
                     plot_file_name=plot_file_name,
                     tests=['XYY', 'WYY', 'WY^Y', 'WYY^', 'WY^Y^'],
                     constant_baseline=0.2475647266285736)


if __name__ == '__main__':

    compas_rocplot()
    compas_squared_barplot()
    compas_zeroone_barplot()
