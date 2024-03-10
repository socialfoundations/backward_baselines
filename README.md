# Backward baselines

This is code to reproduce experients in the paper:

> Hardt and Kim. Is your model predicting the past? ACM EAAMO 2023.

Cite as:

```
@inproceedings{hardtkim2023predicting,
  title={Is Your Model Predicting the Past?},
  author={Kim, Michael P and Hardt, Moritz},
  booktitle={Proc.~$3$rd ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization (EAAMO)},
  pages={1--8},
  year={2023}
}
```

## Step 1: Download and preprocessing the data

To download and preprocess all necessary data files, follow the instructions in each of the following files:

* meps/data/README.md
* sipp/data/README.md
* compas/data/README.md

## Step 2: Plotting the figures from the paper

To produce all figures run:

```
> python meps_experiments.py
> python sipp_experimtents.py
> python compas_experiments.py
```

On first run this will produce json files with results in the `results/` directory and then plot the figures.
