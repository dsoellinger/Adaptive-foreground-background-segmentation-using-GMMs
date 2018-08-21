from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal
import numpy as np


class Gaussian:

    _mean = None
    _covariance = None

    def __init__(self, mean, covariance):
        self._mean = np.array(mean)
        self._covariance = np.array(covariance)

    def mahalanobis_distance_between(self, x):
        gaus =  mahalanobis(x, self._mean, self._covariance)
        return gaus.pdf(x)

    def pdf(self, x):
        return multivariate_normal(self._mean, self._covariance).pdf(x)

    def get_mean(self):
        return self._mean

    def partial_fit(self, x, lr=0.75):
        x = np.array(x)
        roh = lr * self.pdf(x)
        self._mean = (1 - roh) * self._mean + roh * x
        self._covariance = (1-roh) * self._covariance + roh * (x - self._mean).transpose() * (x - self._mean)