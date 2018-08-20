import numpy as np
from scipy.spatial.distance import mahalanobis


class Gaussian:

    _mean = None
    _covariance = None

    def __init__(self, mean, covariance):
        self._mean = mean
        self._covariance = covariance

    def mahalanobis_distance_between(self, point):
        return mahalanobis(point, self._mean, self._covariance)