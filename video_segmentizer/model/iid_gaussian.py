from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal
import numpy as np


class IIDGaussian:

    _mean = None
    _variance = None
    _dim = None

    def __init__(self, mean, variance, dim):
        self._mean = np.array(mean)
        self._variance = variance
        self._dim = dim

    def mahalanobis_distance_between(self, x):
        return mahalanobis(x, self._mean, self.get_covariance())

    def pdf(self, x):
        gaus = multivariate_normal(self._mean, self.get_covariance())
        return gaus.pdf(x)

    def get_mean(self):
        return self._mean

    def get_variance(self):
        return self._variance

    def get_covariance(self):
        return self._variance * np.identity(self._dim)

    def partial_fit(self, x, lr=0.75):
        x = np.array(x)
        roh = lr * self.pdf(x)
        self._mean = (1 - roh) * self._mean + roh * x
        self._variance = (1-roh) * self._variance + roh * (x - self._mean).transpose() * (x - self._mean)
