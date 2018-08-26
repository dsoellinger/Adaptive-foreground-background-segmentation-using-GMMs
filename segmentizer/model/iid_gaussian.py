import math
from numba import jit, float64, jitclass
import numba
import numpy as np


@jit(float64(float64[:], float64, float64[:]), nopython=True)
def _mahalanobis_distance_between(mean, inv_variance, x):

    # DIFF = x-mu
    diff = x-mean

    # A = DIFF^T * COV^-1
    a = diff * inv_variance

    # B = A * (X-MU)
    b = a[0] * diff[0] + a[1] * diff[1] + a[2] * diff[2]

    # MAHALANOBIS = sqrt(B)
    mahalanobis = math.sqrt(b)

    return mahalanobis


@jit(float64(float64[:], float64, float64, float64[:]), nopython=True)
def _pdf(mean, variance, inv_variance, x):

    # DIFF = x-mu
    diff = x-mean

    # A = DIFF^T * COV^-1
    a = [diff[0] * inv_variance, diff[1] * inv_variance, diff[2] * inv_variance]

    # B = A * (X-MU)
    b = a[0] * diff[0] + a[1] * diff[1] + a[2] * diff[2]

    # EXPONENT = -0.5 * B
    exp = -0.5 * b

    # C = (2*pi)^(3/2) * |COV|^(1/2) = (SIGMA^3)^(1/2)
    c = (2 * math.pi) ** (3 / 2) * variance ** (3 / 2)

    # pdf = 1 / C * e ^ EXPONENT
    pdf = 1 / c * math.exp(exp)

    return pdf


@jit(float64[:](float64, float64[:], float64[:]), nopython=True)
def _partial_fit_mean(roh, mean, x):
    return (1 - roh) * mean + roh * x


@jit(float64(float64, float64[:], float64, float64[:]), nopython=True)
def _partial_fit_variance(roh, mean, variance, x):

    # DIFF = X - mu
    diff = x - mean

    # A = DIFF^T * DIFF
    a = np.dot(diff, diff)

    # VAR = (1-roh) * VAR + roh * A
    new_variance = (1 - roh) * variance + roh * a

    return new_variance


spec = [
    ('_mean', float64[:]),               # a simple scalar field
    ('_variance', float64),          # an array field
    ('_sigma', float64),          # an array field
    ('_inv_variance', float64),          # an array field
]


@jitclass(spec)
class IIDGaussian:

    def __init__(self, mean, variance):
        self._mean = mean
        self._variance = variance
        self._sigma = math.sqrt(self._variance)
        self._inv_variance = 1 / variance

    def mahalanobis_distance_between(self, x):

        return _mahalanobis_distance_between(self._mean, self._inv_variance, x)

    def pdf(self, x):

        return _pdf(self._mean, self._variance, self._inv_variance, x)

    def get_mean(self):
        return self._mean

    def get_variance(self):
        return self._variance

    def get_sigma(self):
        return self._sigma

    def partial_fit(self, x, lr):

        roh = lr * self.pdf(x)

        # mu = (1-roh) * mu + roh * X
        new_mean = _partial_fit_mean(roh, self._mean, x)

        new_variance = _partial_fit_variance(roh, self._mean, self._variance, x)

        self._variance = new_variance
        self._sigma = math.sqrt(new_variance)
        self._inv_variance = 1 / new_variance
        self._mean = new_mean


    def __str__(self):
        s = 'IIDGaussian[ Mean: ' + str(self._mean) + ', Sigma: ' + str(self._sigma) + ']'
        return s
