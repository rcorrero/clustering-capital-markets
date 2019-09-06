from thermidor import ClustererSocket
from sklearn.cluster import KMeans

from ..functions import labeled_data_joiner


class KMeansSocket(ClustererSocket):
    '''Class which allows for passing `means_`
    into `KMeans` from `GaussianMixtureModel`
    [or any other object which has a `means_`
    attribute].
       
    Parameters
    ----------
    pca_name : string
        Name of pipeline step from which `index_` is extracted.
        
    gmm_name : string
            Name of pipeline step from which `means_` is extracted.
            
    labels_ : list of strings, optional default=None
        Labels of each point.
        
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
            
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
            
    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
            
    verbose : integer
        Controls the verbosity: the higher, the more messages.
    '''
    def __init__(self, pca_name=None, gmm_name=None, labels_=None,
                 cv=3, n_jobs=None, random_state=None, verbose=False):
        self.pca_name = pca_name
        self.gmm_name = gmm_name
        self.labels_ = labels_
        
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
    def fit(self, X, y=None, **kwargs):
        '''Fits `KMeans` using `means_`
        from `gmm_name` step in pipeline.
        '''
        self.pipeline = returns_pipeline
            
        # Extract `means_` from previous step in pipeline
        self._means = self.pipeline.named_steps[
            self.gmm_name].means_
        
        # Calculate number of clusters
        self.n_clusters = self._means.shape[0]
        
        # Create estimator object
        self.estimator = KMeans(n_clusters=self.n_clusters, 
                                init=self._means, verbose=self.verbose, 
                                random_state=self.random_state, n_jobs=-1)
            
        self.estimator.fit(X)
            
        # Set `labels_` attribute for future steps
        self.labels_ = self.estimator.labels_
            
        return self
    
    def transform(self, X):
        '''Returns X.
            
        Implemented so that this estimator may be used
        as intermediate step in pipeline.
        '''
        # Verify estimator has been fitted
        assert self.labels_ is not None, 'Estimator is not fitted yet.'
        
        # Create dataframe containing dates associated with clusters
        return labeled_data_joiner(X, self.labels_, self.pca_name)
