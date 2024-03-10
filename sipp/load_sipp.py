"""Load preprocessed SIPP data."""

import pandas as pd


def load_sipp(wave_1_file='sipp/data/sipp_2014_wave_1.csv',
              wave_2_file='sipp/data/sipp_2014_wave_2.csv'):
    """Load sipp data from preprocessed csv files."""
    w1 = pd.read_csv(wave_1_file)
    w2 = pd.read_csv(wave_2_file)
    X = w1[w1.columns[5:]]
    y = 1.0*(w2['OPM_RATIO'] >= 3)

    return X, y
