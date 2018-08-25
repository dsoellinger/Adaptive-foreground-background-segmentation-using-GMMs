import unittest
from segmentizer.model import IIDGaussian
import numpy as np


class IDDGaussianTest(unittest.TestCase):

    def test_pdf(self):

        test_cases = [
            # Mean, Variance, x, Expected density
            ([5, 5, 5], 6, [1, 1, 1], 7.912712643729114e-05),
            ([5, 5, 5], 6, [0, 0, 0], 8.33993776829909e-06),
            ([5, 9, 2], 6, [0, 0, 0], 4.5130544270661724e-07)
        ]

        for mean, variance, x, expected_density in test_cases:
            gaussian = IIDGaussian(mean, variance)
            density = gaussian.pdf(x)
            assert density == expected_density

    def test_mahalanobis(self):

        test_cases = [
            # Mean, Variance, x, Expected distance
            ([5, 9, 2], 6, [5, 9, 2], 0),
            ([5, 9, 2], 6, [5, 9, 3], 0.408248290463863),
            ([5, 9, 2], 6, [1, 9, 3], 1.6832508230603462),
            ([5, 9, 1], 2, [1, 8, 3], 3.24037034920393),
        ]

        for mean, variance, x, expected_distance in test_cases:
            gaussian = IIDGaussian(mean, variance)
            distance = gaussian.mahalanobis_distance_between(x)
            assert distance == expected_distance


if __name__ == '__main__':
    unittest.main()