"""Code for preprocessing and loading MEPS HC-216 2019 Full Year Consolidated Data File."""

import pandas as pd


def load_meps(data_file='meps/data/h216.dta', classification=True):
    """Load and preprocess MEPS 2019 data, panel 23 and 24.
    
    See https://meps.ahrq.gov/data_stats/download_data/pufs/h216/h216doc.pdf
    for documentation.
    """
    
    # Load MEPS HC-216 2019 Full Year Consolidated Data File
    meps = pd.read_stata(data_file)

    # replace categorical columns by integer
    cat_cols = meps.select_dtypes(['category']).columns
    meps[cat_cols] = meps[cat_cols].apply(lambda x: x.cat.codes)

    # drop object columns
    meps = meps.drop(meps.select_dtypes(['object']), axis=1)

    # Features classified by MEPS documentation as demographic
    demographic_columns = ['SEX','RACEV1X','RACEV2X','RACEAX','RACEBX','RACEWX','RACETHX',
                           'HISPANX','HISPNCAT','EDUCYR','HIDEG','OTHLGSPK','HWELLSPK','BORNUSA',
                           'WHTLGSPK','YRSINUS']

    def filter_cols(cols, suffix):
        round31_cols = []
        for col in cols:
            if col[-len(suffix):] == suffix:
                round31_cols.append(col)
        return round31_cols

    # Features collected during round 3 of Panel 23 and round 1 of Panel 24.
    # These are the first rounds of Panel 23 and 24 in 2019, respectively.
    round31_columns = filter_cols(meps.columns, '31') + filter_cols(meps.columns, '31X')

    def utilization(row):
        """Measure of health care utilization at the end of the year."""
        return row['OBTOTV19'] + row['OPTOTV19'] + row['ERTOT19'] + row['IPNGTD19'] + row['HHTOTD19']

    meps['TOTEXP19'] = meps.apply(lambda row: utilization(row), axis=1)

    features = round31_columns + demographic_columns
    
    X = meps[features]
    
    if classification:
        # Round target variable around median value to give roughly balanced classes.
        y = 1.0*(meps['TOTEXP19'] > 3)
    else:
        y = meps['TOTEXP19']
    
    return X, y
