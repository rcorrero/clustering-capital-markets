import numpy as np
import pandas as pd


def transform_returns(df, alpha=1.0):
    '''Extracts weighted returns data from
    CRSP raw data.
    
    Parameters
    ----------
    df : array-like
    
    alpha : float, optional default=1.0
        Real number to multiply weighted returns by.
        
    Returns
    -------
    Pandas dataframe containing weighted returns.
    '''
    # Pivot the raw data since assets are stacked
    df = pd.pivot(df, 
                  values = ['BIDLO', 'ASKHI', 'VOL', 'RETX'], 
                  index='date', columns = 'PERMNO'
                 )
    
    # Infers objects; those that can't be inferred are
    # converted to `NaN`
    df = df.convert_objects(convert_numeric=True)
    
    # Replace all missing entries with '0'
    df.replace('NaN',0)
    
    # Functions for `vectorize` to apply elementwise to
    # pairs of elements from columns
    average = lambda x, y : ((x + y) / 2.0)
    
    multiply = lambda x, y : (x * y)
    
    divide = lambda x, y : (x / y)
    
    permnos = df['RETX'].columns.tolist()
    
    for permno in permnos:
        # Step 1 - Calculate midprice
        df[('MIDPRCE', permno)] = np.vectorize(average)(df[('ASKHI', permno)],
                                                        df[('BIDLO', permno)])
        
        # Step 2 - Calculate midprice dollar-value traded
        df[('DLRTRDED', permno)] = np.vectorize(multiply)(df[('MIDPRCE', permno)],
                                                          df[('VOL', permno)])
        
    # Sum total dollar-value traded by day
    df['TTLDLRTRDED'] = df['DLRTRDED'].sum(axis=1)
    
    # Drop indexes containing '0' entries in 'TTLDLRTRDED'
    # These are days with no market activity
    df.drop(df.index[df['TTLDLRTRDED'] == 0], inplace=True)
    
    for permno in permnos:
        # Step 3 - Calculate the percentage of total dollar-value traded
        df[('PRCNTDLRTRDED', permno)] = np.vectorize(divide)(df[('DLRTRDED', permno)], 
                                                             df['TTLDLRTRDED'])
        
        # Step 4 - Calculate weighted return
        df[('WGHTEDRETX', permno)] = alpha * np.vectorize(multiply)(
            df[('PRCNTDLRTRDED', permno)], df[('RETX', permno)])
    
    # Sort dataframe by PERMNO
    df = df.sort_index(axis=1, level=1)
    
    # Only return weighted returns
    X = df['WGHTEDRETX']
    
    return X
