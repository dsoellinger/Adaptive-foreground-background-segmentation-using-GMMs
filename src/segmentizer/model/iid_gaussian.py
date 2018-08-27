import math
from numba import float64, jitclass

spec = [
    ('_mean', float64[:]),
    ('_variance', float64),
    ('_sigma', float64),
    ('_inv_variance', float64)
]


@jitclass(spec)
class IIDGaussian:

    def __init__(self, mean, variance):
        self._mean = mean
        self._variance = variance
        self._sigma = math.sqrt(self._variance)
        self._inv_variance = 1 / variance

    def mahalanobis_distance_between(self, x):

        # DIFF = x-mu
        diff = x - self._mean

        # A = DIFF^T * COV^-1
        a = diff * self._inv_variance

        # B = A * (X-MU)
        b = a[0] * diff[0] + a[1] * diff[1] + a[2] * diff[2]

        # MAHALANOBIS = sqrt(B)
        mahalanobis = math.sqrt(b)

        return mahalanobis

    def pdf(self, x):

        # DIFF = x-mu
        diff = x - self._mean

        # A = DIFF^T * COV^-1
        a = diff * self._inv_variance

        # B = A * (X-MU)
        b = a[0] * diff[0] + a[1] * diff[1] + a[2] * diff[2]

        # EXPONENT = -0.5 * B
        exp = -0.5 * b

        # C = (2*pi)^(3/2) * |COV|^(1/2) = (VARIANCE^3)^(1/2)
        c = math.pow(2 * math.pi * self._variance, 3/2)

        # pdf = 1 / C * e ^ EXPONENT
        pdf = 1 / c * math.exp(exp)

        return pdf

    def get_mean(self):
        return self._mean

    def get_variance(self):
        return self._variance

    def get_sigma(self):
        return self._sigma

    def partial_fit(self, x, lr):
        roh = lr * self.pdf(x)

        # mu = (1-roh) * mu + roh * X
        new_mean = (1 - roh) * self._mean + roh * x

        # DIFF = X - mu
        diff = x - self._mean

        # A = DIFF^T * DIFF
        a = diff[0]* diff[0] + diff[1]*diff[1] + diff[2]*diff[2]

        # VAR = (1-roh) * VAR + roh * A
        new_variance = (1 - roh) * self._variance + roh * a

        self._variance = new_variance
        self._sigma = math.sqrt(new_variance)
        self._inv_variance = 1 / new_variance
        self._mean = new_mean

    def __str__(self):
        s = 'IIDGaussian[ Mean: ' + str(self._mean) + ', Sigma: ' + str(self._sigma) + ']'
        return s
