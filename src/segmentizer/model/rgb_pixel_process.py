from .iid_gaussian import IIDGaussian
import numpy as np


class RGBPixelProcess:

    def __init__(self, n_clusters):
        self._n_clusters = n_clusters
        self._init_mixture()

    def _init_mixture(self):
        self._mixture = []
        for _ in range(self._n_clusters):
            self._mixture.append((0, IIDGaussian(np.array([0.,0.,0.]), 1.)))

    def _get_background_distributions(self, t=0.7):

        sum_weight = 0.0
        background_distributions = []

        for weight, gaussian in self._mixture:
            sum_weight += weight
            background_distributions.append(gaussian)
            if sum_weight > t:
                return background_distributions

        return []

    def _normalize_weights(self, total_weight):
        self._mixture = [(w/total_weight, gaus) for w, gaus in self._mixture]

    def fit(self, x, init_weight, init_variance, lr):

        best_matching_gaussian_idx = None
        total_weight = 0.0

        for i, (w, gaus) in enumerate(self._mixture):

            # One the first (best) match gets the "treat"
            if gaus.mahalanobis_distance_between(x) < 2.5 and best_matching_gaussian_idx is None:
                gaus.partial_fit(x, lr)
                new_weight = (1 - lr) * w + lr
                best_matching_gaussian_idx = i
            else:
                new_weight = (1 - lr) * w

            self._mixture[i] = (new_weight, gaus)
            total_weight += new_weight

        # Replace last gaussian (w/sigma) if no match was found
        if best_matching_gaussian_idx is None:
            total_weight = total_weight - self._mixture[self._n_clusters-1][0] + init_weight
            self._mixture[self._n_clusters-1] = (init_weight, IIDGaussian(x.astype('float64'), init_variance))

        self._normalize_weights(total_weight)

        self._mixture.sort(key=lambda x: x[0] / x[1].get_sigma(), reverse=True)

    def is_background_pixel(self, x):

        background_dist = self._get_background_distributions()
        for gaus in background_dist:
            if gaus.mahalanobis_distance_between(x) < 2.5:
                return True
        return False

    def __str__(self):
        s = 'RGBPixelProcess[' + ','.join(map(lambda x: 'w:' + str(x[0]) + ', gaus:' + str(x[1]),self._mixture)) + ']'
        return s
