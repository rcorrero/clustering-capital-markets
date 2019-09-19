'''
clustering_capital_markets
=================================
This is a Python module containing the source code for the project of the same name.

In this project we study capital as it flows through the American equity markets.
Using daily returns and volume data we develop a method to identify regimes in the 
markets over time, regimes which characterize much of the markets' behavior. By
segmenting the data we analyze the causes of market behavior associated with 
a given regime.

The main intuition underpinning this project is that the financial markets, as the 
conduits of capital, act as "encoders" recording the behavior of capital in the face 
of new information. This information is encoded in the price histories of the 
instruments traded on the markets. 
'''

# Import objects
from .classes import GMMSocketCV
from .classes import KMeansSocket

from .functions import gmm_dist
from .functions import labeled_data_joiner
from .functions import pca_dist
from .functions import transform_returns
