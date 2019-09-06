clustering-capital-market-returns &mdash; Richard Correro
==============================
This is a machine learning project created using [Sci-kit Learn](https://github.com/scikit-learn/scikit-learn) and [thermidor](https://github.com/rcorrero/thermidor).

In this project we study capital as it flows through the American equity markets. Using daily returns and volume data we develop a method to identify regimes in the markets over time, regimes which characterize much of the markets' behavior. By segmenting the data we analyze the causes of market behavior associated with a given regime.

The main intuition underpinning this project is that the financial markets, as the conduits of capital, act as "encoders" recording the behavior of capital in the face of new information. This information is encoded in the price histories of the instruments traded on the markets. 

# Data
In this project we use returns, price, and volume data from a universe of equities traded on exchanges in the United States. This data is provided by the [Center for Research in Security Prices](http://www.crsp.com/) and obtained through [Wharton Research Data Services](https://wrds-web.wharton.upenn.edu/wrds/). We access this data through our institution, and we do not have the rights to publish it. Because of this the `data` folder in our local directory is excluded from this repository. 

The data required for this project is available through other sources, and if you need help obtaining data then feel free to contact Richard Correro.

# thermidor
While developing this standard framework for data science projects we've created and utilized several functions and classes to streamline and simplify the process of pipeline construction. We've created a python module named [thermidor](https://github.com/rcorrero/thermidor) where these objects may be found. 

------------
# Organization
```
.
├── LICENSE
├── README.md
├── clustering-capital-market-returns
│   ├── __init__.py
│   ├── classes
│   │   ├── __init__.py
│   │   ├── gmm_socket_cv.py
│   │   └── k_means_socket.py
│   └── functions
│       ├── __init__.py
│       ├── gmm_dist.py
│       ├── labeled_data_joiner.py
│       ├── pca_dist.py
│       └── transform_returns.py
└── notebooks
    └── 0.7.3-returns-pipeline-analysis.ipynb
    
``` 
------------
# Dependencies
- Python
- Pandas
- NumPy
- SciPy
- [thermidor](https://github.com/rcorrero/thermidor)
------------
Created by Richard Correro in 2019. Contact me at rcorrero at stanford dot edu
