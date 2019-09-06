import pandas as pd
from scipy.stats import randint


def gmm_dist(df, proportion=.1):
    '''Creates scipy randint distribution
    with max=(n_samples * proportion) of df.
    
    Parameters
    ----------
    df : array-like

    proportion : float, optional default=.1
        Determines the maximum of the range for the distribution.
    
    Returns
    -------
    scipy.stats object
    '''
    dist_max = df.shape[0] * proportion
    
    return randint(1, dist_max)
