"""Various plotting functions for backward baselines."""

import numpy as np
import matplotlib.pyplot as plt


test_labels = { 'XYY' : r'$XYY$',
                'WYY' : r'$WYY$',
                'WY^Y' : r'$W\hat YY$',
                'WYY^' : r'$WY\hat Y$',
                'WY^Y^' : r'$W\hat Y\hat Y$' }

test_colors = { 'XYY' : 'steelblue',
                'WYY' : 'red',
                'WY^Y' : 'orangered',
                'WYY^' : 'darkorange',
                'WY^Y^' : 'orange' }


def make_XYY_consistent(results):
    """Make sure all feature settings have same XYY plot."""
    features = list(results.keys())
    models = list(results[features[0]].keys())
    for model in models:
        for feature in features:
            results[feature][model]['XYY'] = results[features[0]][model]['XYY']
    return results



def roc_plot_results(results, model_names=None,
                     plot_file_name='rocplot.pdf',
                     tests=['XYY', 'WYY', 'WYY^']):
    """Plot ROC curves from results."""

    results = make_XYY_consistent(results)

    plt.rcParams["font.family"] = "Times New Roman"

    features = list(results.keys())
    models = list(results[features[0]].keys())

    n_rows = len(models)
    n_cols = len(features)

    x_size = 2.5 * n_cols
    y_size = 2 * n_rows
    if n_rows == 1:
        y_size = 3
    fig, axs = plt.subplots(n_rows, n_cols, 
                            sharex=True, sharey=True, figsize=(x_size, y_size))
    
    shape = (n_rows, n_cols)
    axs = np.reshape(axs, shape)
    
    for (i, model) in enumerate(models):
        for (j, feature) in enumerate(features):
            ax = axs[i, j]
            
            if i == 0:
                ax.set_title(feature, fontsize=14)
            if i == n_rows - 1:
                ax.set_xlabel('false positive rate', fontsize=14)
            if j == 0:
                if model_names:
                    model_name = model_names[i]
                else:
                    model_name = str(model)
                ax.set_ylabel(model_name+'\n true positive rate', fontsize=14)

            ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100),
                    color='black', alpha=0.5, linestyle='--')
            for test in tests:
                fprs, tprs, _ = results[feature][model][test][0]
                ax.plot(fprs, tprs, color=test_colors[test], alpha=0.7,
                        label=test_labels[test], linewidth=1)
                for (fprs, tprs, _) in results[feature][model][test][1:]:
                    ax.plot(fprs, tprs, color=test_colors[test],
                            alpha=0.1, linewidth=2.5)
            
            ax.legend()

    plt.tight_layout()
    plt.savefig(plot_file_name)
                    

def bar_plot_results(results, model_names=None, loss_name='zero-one loss', 
                     plot_file_name='barplot.pdf',
                     tests=['XYY', 'WYY', 'WYY^'],
                     constant_baseline=None):
    """Grid of bar plots. One row for each feature setting. One column for each model."""

    results = make_XYY_consistent(results)

    plt.rcParams["font.family"] = "Times New Roman"

    features = list(results.keys())
    models = list(results[features[0]].keys())

    n_rows = len(models)
    n_cols = len(features)

    x_size = 2.5 * n_cols
    y_size = 2 * n_rows
    fig, axs = plt.subplots(n_rows, n_cols, 
                            sharex=True, sharey=True, figsize=(x_size, y_size))
    
    shape = (n_rows, n_cols)
    axs = np.reshape(axs, shape)
    
    for (i, model) in enumerate(models):
        for (j, feature) in enumerate(features):
            ax = axs[i, j]
            
            mean_scores = [results[feature][model][test]['mean'] for test in tests]
            std_scores = [results[feature][model][test]['std'] for test in tests]
            
            if i == 0:
                ax.set_title(feature, fontsize=14)
            if j == 0:
                if model_names:
                    model_name = model_names[i]
                else:
                    model_name = str(model)
                ax.set_ylabel(model_name + '\n' + loss_name, fontsize=14)
            ax.set_yticks([0,0.25,0.5])
            ax.set_ylim([0,0.5])
            
            ax.bar([test_labels[test] for test in tests],
                   mean_scores, yerr=std_scores, 
                   color=[test_colors[test] for test in tests],
                   alpha=0.5, ecolor='black', capsize=10)

            if constant_baseline:
                ax.axhline(y=constant_baseline, color='black', alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.savefig(plot_file_name)

