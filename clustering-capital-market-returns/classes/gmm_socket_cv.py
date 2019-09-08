from thermidor import EstimatorSocketCV


class GMMSocketCV(EstimatorSocketCV):
    '''Class which fits an estimator with  an unknown parameter
    using a Sci-kit Learn cross-validator.
    
    Takes a distribution function as a parameter. 
    This distribution is passed to `cross_val` 
    as the distribution for `param_name` in `estimator`.
    This allows for dyanmic distribution creation. For
    example, when the dimensions of X are changed by a 
    Sci-kit Learn `transformer` in a pipeline, 
    `EstimatorSocketCV` allows for parameter distributions
    which take the transformed X as an input. 
    
    Parameters
    -----------
    estimator : Sci-kit Learn estimator class
        Estimator to be fit using cross-validation.
        
    param_name: str
        Name of parameter to be chosen by crosw-validation.
        
     dist_func : calleable 
        Must take X as sole parameter and return scipy.stats 
        distribution or object that implements `rvs` method.
        This object is passed as parameter distribution to 
        `cross_val`.
        
    cross_val : Sci-kit Learn cross-validator object
        Cross-validator used to fit `estimator`.
        
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

    cv_random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        This values is passed to `cross_val`.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    estimator_random_state : int, RandomState instance or None, optional, 
                             default=None
        This value is passed to `estimator`.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
        
    verbose : integer
        Controls the verbosity: the higher, the more messages.
    '''
    def fit(self, X, y=None, **kwargs):
        '''Fits estimator using `cross_val`.

        Stores `means_ as attribute for pipeline.
        '''
        # Distribution passed to `cross_val`
        self.dist = {
            self.param_name : self.dist_func(X)
        }

        # Create cross-validator object
        self.model_selector = self.cross_val(self.estimator(
            random_state=self.estimator_random_state), 
            self.dist, cv=self.cv, n_jobs=self.n_jobs,
            random_state=self.cv_random_state, verbose=self.verbose)
        
        self.model_selector.fit(X, y, **kwargs)
        
        # Store `best_estimator_`
        self.best_estimator_ = self.model_selector.best_estimator_
        
        # Store `means_` for use in pipeline
        self.means_ = self.best_estimator_.means_
        
        return self
    
    def transform(self, X):
        '''Returns X.
        
        Implemented so that this estimator may be used
        as intermediate step in pipeline.
        '''
        return X
