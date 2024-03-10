"""Coad to load and preprocess ProPublica compas data."""

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class CompasModel:
    """Wrapper around compas scores."""
    
    def __init__(self, data):
        self._data = data
        
    def __str__(self):
        return "Compas score"
    
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        return self._data.iloc[X.index]['compas_predictions']


class CompasRegressor:
    """Wrapper around compas scores."""
    
    def __init__(self, data):
        self._data = data
        
    def __str__(self):
        return "Compas score wrapper"
    
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        return self._data.iloc[X.index]['compas_regressions']


def load_compas(data_file='compas/data/compas-scores-two-years.csv'):
    """Preprocess and load compas data"""

    data = pd.read_csv(data_file) 
    outcomes = data['two_year_recid']
    raceint = {'Other' : 0,
               'African-American' : 1,
               'Hispanic' : 2,
               'Caucasian' : 3,
               'Asian' : 4 ,
               'Native American' : 5}
    data['raceint'] = data['race'].map(raceint)
    
    gb = GradientBoostingClassifier()
    gb.fit(data[['decile_score']], data['two_year_recid'])
    data['compas_predictions'] = gb.predict(data[['decile_score']])

    scaled_regression = make_pipeline(StandardScaler(), LogisticRegression())
    scaled_regression.fit(data[['decile_score']], data['two_year_recid'])
    predictions = scaled_regression.predict_proba(data[['decile_score']])[:,1]
    data['compas_regressions'] = predictions
    
    return data, outcomes
