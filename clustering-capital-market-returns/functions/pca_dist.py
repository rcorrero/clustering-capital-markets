import pandas as pd
from scipy.stats import randint


def pca_dist(df, proportion=.8):
    '''Creates scipy randint distribution
    with max=min(n_samples, n_features) of
    df.
    
    Parameters
    ----------
    df : array-like

    proportion : float, optional default=.8
        Determines the maximum of the range for the distribution.
        Defaults to .8 since 5-fold cross-validation uses at most 
        80 percent of the data.
    
    Returns
    -------
    scipy.stats object
    '''
    n_samples = df.shape[0] * proportion
    
    n_features = df.shape[1]
    
    dist_max = min(n_samples, n_features)
    
    return randint(1, dist_max)
