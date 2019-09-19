from setuptools import find_packages, setup

setup(
    name='clustering_capital_markets',
    packages=find_packages(),
    version='0.1',
    description='clustering_capital_markets is a python module containing the source code from the project of the same name. In this project we study capital as it flows through the American equity markets. Using daily returns and volume data we develop a method to identify regimes in the markets over time, regimes which characterize much of the markets behavior. By segmenting the data we analyze the causes of market behavior associated with a given regime.',
    author='Richard Correro',
    license='BSD-3-Clause',
)
