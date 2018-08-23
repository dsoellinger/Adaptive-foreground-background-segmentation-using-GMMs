from scipy.spatial.distance import mahalanobis, cdist
from scipy.stats import multivariate_normal
import numpy as np


class IIDGaussian:

    _mean = None
    _variance = None
    _dim = None
    _covariance = None
    _inv_coveriance = None

    def __init__(self, mean, variance, dim):
        self._mean = np.array(mean)
        self._variance = variance
        self._covariance = variance * np.identity(dim)
        self._inv_coveriance = np.linalg.inv(self._covariance)
        self._dim = dim

    def mahalanobis_distance_between(self, x):
        #return cdist(x, self._mean, metric='mahalanobis', VI=self._inv_coveriance)
        return mahalanobis(x, self._mean, self._inv_coveriance)

    def pdf(self, x):
        gaus = multivariate_normal(self._mean, self._covariance)
        return gaus.pdf(x)

    def get_mean(self):
        return self._mean

    def get_variance(self):
        return self._variance

    def get_covariance(self):
        return self._covariance

    def partial_fit(self, x, lr=0.75):
        roh = lr * self.pdf(x)
        self._mean = (1 - roh) * self._mean + roh * x
        self._variance = (1-roh) * self._variance + roh * np.matmul(np.matrix(x) - self._mean,(np.matrix(x) - self._mean).transpose())[0,0]