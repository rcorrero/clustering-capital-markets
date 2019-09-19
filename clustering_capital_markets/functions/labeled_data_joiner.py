import pandas as pd


def labeled_data_joiner(df, labels, step_name):
    '''Creates a Pandas datafrme associating
    dates with the clusters to which they belong
    
    These clusters should be mutually-exclusive 
    partitions of the data space.
    
    Parameters
    ----------
    df : array-like
    
    labels : 
        Labels of each point
    '''
    # Extract `index_` from 'decomposer'
    dates = returns_pipeline.named_steps[step_name].index_

    # Create dataframe
    labeled_data = pd.DataFrame({'dates': dates, 
                                 'cluster': labels})
    
    return labeled_data
