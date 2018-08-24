import numpy as np
import math


class IIDGaussian:

    _mean = None
    _variance = None
    _inv_variance = None
    _sigma = None

    def __init__(self, mean, variance):
        self._mean = np.array(mean)
        self._variance = variance
        self._sigma = math.sqrt(self._variance)
        self._inv_variance = 1 / variance

    def mahalanobis_distance_between(self, x):

        # DIFF = x-mu
        diff = x - self._mean

        # A = DIFF^T * COV^-1
        a = diff * self._inv_variance

        # B = A * (X-MU)
        b = np.dot(a, diff)

        # MAHALANOBIS = sqrt(B)
        mahalanobis = math.sqrt(b)

        return mahalanobis

    def pdf(self, x):

        # DIFF = x-mu
        diff = x - self._mean

        # A = DIFF^T * COV^-1
        a = diff * self._inv_variance

        # B = A * (X-MU)
        b = np.dot(a, diff)

        # EXPONENT = -0.5 * B
        exp = -0.5 * b

        # C = (2*pi)^(3/2) * |COV|^(1/2) = (SIGMA^3)^(1/2)
        c = (2*math.pi)**(3/2) * self._variance ** (3/2)

        # pdf = 1 / C * e ^ EXPONENT
        pdf = 1/c * math.exp(exp)

        return pdf

    def get_mean(self):
        return self._mean

    def get_variance(self):
        return self._variance

    def get_sigma(self):
        return self._sigma

    def partial_fit(self, x, lr=0.75):

        roh = lr * self.pdf(x)

        # mu = (1-roh) * mu + roh * X
        self._mean = (1 - roh) * self._mean + roh * x

        # DIFF = X - mu
        diff = x - self._mean

        # A = DIFF^T * DIFF
        a = np.dot(diff, diff)

        # VAR = (1-roh) * VAR + roh * A
        self._variance = (1-roh) * self._variance + roh * a

        self._sigma = math.sqrt(self._variance)
        self._inv_variance = 1 / self._variance

